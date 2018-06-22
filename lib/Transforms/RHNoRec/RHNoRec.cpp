//===- RHNoRec.cpp - Hybrid Transactional Memory --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements two versions of the LLVM "Hello World" pass described
// in docs/WritingAnLLVMPass.html
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <array>
#include <cstdio>
#include <map>
#include <queue>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "rhnorec"

#define DEFAULT_MAX_READS         256
#define DEFAULT_EXPECTED_LENGTH   256

#define CTX_JUMP_BUFFER          0
#define CTX_IS_TX_VERSION        1
#define CTX_MAX_READS            2
#define CTX_EXPECTED_LENGTH      3
#define CTX_PREFIX_READS         4
#define CTX_IS_RH_PREFIX_ACTIVE  5
#define CTX_IS_RH_ACTIVE         6
#define CTX_IS_WRITE_DETECTED    7
#define CTX_IS_READONLY          8

STATISTIC(RHNoRecCounter, "Counts number of functions visited");

namespace {
    const char *transform_begin = "__defrts_hybrid_xbegin";
    const char *transform_end = "__defrts_hybrid_xend";

    const char *frameaddress = "llvm.frameaddress";
    const char *stacksave = "llvm.stacksave";
    const char *llvm_setjmp = "llvm.eh.sjlj.setjmp";

    const char *fast_path_start    = "__rhnorec_fast_path_start";
    const char *slow_path_start    = "__rhnorec_slow_path_start";
    const char *slow_path_write_64 = "__rhnorec_slow_path_write_64";
    const char *slow_path_write_32 = "__rhnorec_slow_path_write_32";
    const char *slow_path_write_16 = "__rhnorec_slow_path_write_16";
    const char *slow_path_write_8  = "__rhnorec_slow_path_write_8";
    const char *slow_path_read_64  = "__rhnorec_slow_path_read_64";
    const char *slow_path_read_32  = "__rhnorec_slow_path_read_32";
    const char *slow_path_read_16  = "__rhnorec_slow_path_read_16";
    const char *slow_path_read_8   = "__rhnorec_slow_path_read_8";
    const char *slow_path_commit   = "__rhnorec_mixed_slow_path_commit";

    StructType *rhnorec_ctx_type;

    struct rhnorec_ctx_t
    {
        Value *ctx_var;
        Type *i1;
        Type *i16;
        Type *i64;
    };

    void ctx_write (IRBuilder<> &bldr,
                    rhnorec_ctx_t &rhnorec_ctx,
                    int member_number,
                    Value *val)
    {
        Value *dest = bldr.CreateStructGEP(rhnorec_ctx_type,
                                           rhnorec_ctx.ctx_var,
                                           member_number);
        bldr.CreateStore(val, dest);
    }

    Function *get_fast_path_start (Module *mdl, LLVMContext &ctx)
    {
        Function *ret = mdl->getFunction(fast_path_start);
        if (ret) return ret;
        FunctionType *fty = FunctionType::get(Type::getInt1Ty(ctx), false);
        return
            dyn_cast<Function>(mdl->getOrInsertFunction(fast_path_start, fty));
    }

    Function *get_slow_path_start (Module *mdl, LLVMContext &ctx)
    {
        Function *ret = mdl->getFunction(slow_path_start);
        if (ret) return ret;
        Type *ptr_to_ctx = PointerType::get(rhnorec_ctx_type, 0);
        FunctionType *fty = FunctionType::get(Type::getVoidTy(ctx),
                                              ArrayRef<Type*>(ptr_to_ctx),
                                              false);
        return
            dyn_cast<Function>(mdl->getOrInsertFunction(slow_path_start, fty));
    }

    Function *get_frameaddress (Module *mdl, LLVMContext &ctx)
    {
        Function *ret = mdl->getFunction(frameaddress);
        if (ret) return ret;
        Type *retty = PointerType::get(Type::getInt8Ty(ctx), 0);
        std::array<Type*, 1> params = { Type::getInt32Ty(ctx) };
        FunctionType *fty = FunctionType::get(retty,
                                              ArrayRef<Type*>(params),
                                              false);
        return
            dyn_cast<Function>(mdl->getOrInsertFunction(frameaddress, fty));
    }

    Function *get_stacksave (Module *mdl, LLVMContext &ctx)
    {
        Function *ret = mdl->getFunction(stacksave);
        if (ret) return ret;
        Type *retty = PointerType::get(Type::getInt8Ty(ctx), 0);
        FunctionType *fty = FunctionType::get(retty, ArrayRef<Type*>(), false);
        return dyn_cast<Function>(mdl->getOrInsertFunction(stacksave, fty));
    }

    Function *get_setjmp (Module *mdl, LLVMContext &ctx)
    {
        Function *ret = mdl->getFunction(llvm_setjmp);
        if (ret) return ret;
        Type *param = PointerType::get(Type::getInt8Ty(ctx), 0);
        FunctionType *fty = FunctionType::get(Type::getInt32Ty(ctx),
                                              ArrayRef<Type*>(param),
                                              false);
        return dyn_cast<Function>(mdl->getOrInsertFunction(llvm_setjmp, fty));
    }

    Function *get_slow_store (Module *mdl, LLVMContext &ctx, unsigned int sz)
    {
        const char *name;

        switch (sz) {
        case 64: name = slow_path_write_64; break;
        case 32: name = slow_path_write_32; break;
        case 16: name = slow_path_write_16; break;
        case 8:  name = slow_path_write_8;  break;
        default:
            std::fprintf(stderr, "rhnorec: unsupported store size %u\n", sz);
            abort();
        }

        Function *ret = mdl->getFunction(name);
        if (ret) return ret;
        Type *ptr_to_ctx = PointerType::get(rhnorec_ctx_type, 0);
        Type *sized_int = Type::getIntNTy(ctx, sz);
        std::array<Type*, 3> params =
            { ptr_to_ctx, PointerType::get(sized_int, 0), sized_int };
        FunctionType *fty = FunctionType::get(Type::getVoidTy(ctx),
                                              ArrayRef<Type*>(params),
                                              false);
        return dyn_cast<Function>(mdl->getOrInsertFunction(name, fty));
    }

    Function *get_slow_load (Module *mdl, LLVMContext &ctx, unsigned int sz)
    {
        const char *name;

        switch (sz) {
        case 64: name = slow_path_read_64; break;
        case 32: name = slow_path_read_32; break;
        case 16: name = slow_path_read_16; break;
        case 8:  name = slow_path_read_8;  break;
        default:
            std::fprintf(stderr, "rhnorec: unsupported load size %u\n", sz);
            abort();
        }

        Function *ret = mdl->getFunction(name);
        if (ret) return ret;
        Type *ptr_to_ctx = PointerType::get(rhnorec_ctx_type, 0);
        Type *sized_int = Type::getIntNTy(ctx, sz);
        std::array<Type*, 2> params =
            { ptr_to_ctx, PointerType::get(sized_int, 0) };
        FunctionType *fty = FunctionType::get(sized_int,
                                              ArrayRef<Type*>(params),
                                              false);
        return dyn_cast<Function>(mdl->getOrInsertFunction(name, fty));
    }

    Function *get_slow_path_commit (Module *mdl, LLVMContext &ctx)
    {
        Function *ret = mdl->getFunction(slow_path_commit);
        if (ret) return ret;
        Type *ptr_to_ctx = PointerType::get(rhnorec_ctx_type, 0);
        std::array<Type*, 1> params = { ptr_to_ctx };
        FunctionType *fty = FunctionType::get(Type::getVoidTy(ctx),
                                              ArrayRef<Type*>(params),
                                              false);
        return
            dyn_cast<Function>(mdl->getOrInsertFunction(slow_path_commit,
                                                        fty));
    }

    /** Return true iff fn needs an RHNoRec transformation.
     */
    bool needs_transform (Function &fn)
    {
        for (BasicBlock &bb : fn) {
            for (Instruction &inst : bb) {
                CallInst *call = dyn_cast<CallInst>(&inst);
                if (!call) continue;
                if (call->getCalledFunction()->getName() == transform_begin) {
                    return true;
                }
            }
        }
        return false;
    }

    /** RHNoRec requires a whole bunch of contextual data stored in the frame.
     *  Prepend allocas for these variables to the entry block.
     */
    rhnorec_ctx_t prepend_context_struct (Module *mdl,
                                          LLVMContext &ctx,
                                          BasicBlock &entry)
    {
        rhnorec_ctx_t ret;

        Type *i1  = Type::getInt1Ty(ctx);
        Type *i8  = Type::getInt8Ty(ctx);
        Type *i16 = Type::getInt16Ty(ctx);
        Type *i64 = Type::getInt64Ty(ctx);

        // Add a bunch of alloca's for context variables.
        IRBuilder<> bldr(ctx);
        bldr.SetInsertPoint(&entry.front());

        if (nullptr == rhnorec_ctx_type) {
            /*
              typedef __rhnorec_ctx =
                  { jump_buffer         [20]*void,
                    is_tx_version       u64,
                    max_reads           u16,
                    expected_length     u16,
                    prefix_reads        u16,
                    is_rh_prefix_active bool,
                    is_rh_active        bool,
                    is_write_detected   bool,
                    is_readonly         bool
                  };
            */

            std::array<Type*, 9> members =
                { ArrayType::get(PointerType::get(i8, 0), 20), 
                  i64, i16, i16, i16, i1, i1, i1, i1
                };
            rhnorec_ctx_type =
                StructType::create(ctx,
                                   ArrayRef<Type*>(members),
                                   "rhnorec.ctx_type");
        }

        ret.ctx_var =
            bldr.CreateAlloca(rhnorec_ctx_type, nullptr, "rhnorec.ctx");

        ret.i1 = i1;
        ret.i16 = i16;
        ret.i64 = i64;

        return ret;
    }

    /** Given the basic block in which the transaction begins and the actual
     *  instruction of the call, perform the transformation needed to convert
     *  it to an RHNoRec hybrid transaction.
     */
    void do_transform (Module *mdl,
                       LLVMContext &ctx,
                       rhnorec_ctx_t &rhnorec_ctx,
                       BasicBlock &pre_bb,
                       Instruction &inst)
    {
        IRBuilder<> bldr(ctx);
        Function *fcn = pre_bb.getParent();
        BasicBlock *xbegin = &pre_bb;
        if (&pre_bb.front() != &inst) {
            // There are instructions before the begin-transaction call.  They
            // should get separated away since we'll re-enter the begin block
            // every time a transaction aborts.
            xbegin = SplitBlock(&pre_bb, &inst);
        }

        // Insert a call to the RHNoRec start procedure which will return
        // whether to proceed along the fast path or slow path.
        inst.eraseFromParent();
        bldr.SetInsertPoint(xbegin, xbegin->begin());
        Value *is_fast = bldr.CreateCall(get_fast_path_start(mdl, ctx),
                                         ArrayRef<Value*>(),
                                         "is_fast");
        BasicBlock *fast_path = SplitBlock(xbegin, &*(bldr.GetInsertPoint()));
        BasicBlock *slow_path_init =
            BasicBlock::Create(ctx,
                               fast_path->getName() + ".slow_init",
                               fcn);
        BasicBlock *slow_path =
            BasicBlock::Create(ctx,
                               fast_path->getName() + ".slow",
                               fcn);
        fast_path->setName(fast_path->getName() + ".fast");
        xbegin->back().eraseFromParent();
        bldr.SetInsertPoint(xbegin);
        bldr.CreateCondBr(is_fast, fast_path, slow_path_init);

        // Start the slow path.
        bldr.SetInsertPoint(slow_path_init);

        // setjmp() as a place to jump back to when the transaction has to be
        // reset manually.  Ignore the return value.
        Value *zero = ConstantInt::get(Type::getInt32Ty(ctx), 0);
        Value *two = ConstantInt::get(Type::getInt32Ty(ctx), 2);

        Value *jump_buf = bldr.CreateStructGEP(rhnorec_ctx_type,
                                               rhnorec_ctx.ctx_var,
                                               CTX_JUMP_BUFFER,
                                               "rh.jumpbuf");
        Value *bp = bldr.CreateCall(get_frameaddress(mdl, ctx),
                                    ArrayRef<Value*>(zero));
        std::array<Value*, 2> idx = { zero, zero };
        Value *bppos = bldr.CreateInBoundsGEP(jump_buf,
                                              ArrayRef<Value*>(idx),
                                              "rh.bp");
        bldr.CreateStore(bp, bppos);
        Value *sp = bldr.CreateCall(get_stacksave(mdl, ctx),
                                    ArrayRef<Value*>(),
                                    "rh.sp");
        idx = { zero, two };
        Value *sppos = bldr.CreateInBoundsGEP(jump_buf,
                                              ArrayRef<Value*>(idx));
        bldr.CreateStore(sp, sppos);
        Value *vp_jump_buf =
            bldr.CreateBitCast(jump_buf,
                               PointerType::get(Type::getInt8Ty(ctx), 0));
        bldr.CreateCall(get_setjmp(mdl, ctx), ArrayRef<Value*>(vp_jump_buf));

        // Initialize the context.
        // FIXME: Necessary?
        ctx_write(bldr, rhnorec_ctx, CTX_IS_TX_VERSION,
                  ConstantInt::get(rhnorec_ctx.i64, 0));
        ctx_write(bldr, rhnorec_ctx, CTX_MAX_READS,
                  ConstantInt::get(rhnorec_ctx.i16, DEFAULT_MAX_READS));
        ctx_write(bldr, rhnorec_ctx, CTX_EXPECTED_LENGTH,
                  ConstantInt::get(rhnorec_ctx.i16, DEFAULT_EXPECTED_LENGTH));
        ctx_write(bldr, rhnorec_ctx, CTX_PREFIX_READS,
                  ConstantInt::get(rhnorec_ctx.i16, 0));
        ctx_write(bldr, rhnorec_ctx, CTX_IS_RH_PREFIX_ACTIVE,
                  ConstantInt::get(rhnorec_ctx.i1, 0));
        ctx_write(bldr, rhnorec_ctx, CTX_IS_RH_ACTIVE,
                  ConstantInt::get(rhnorec_ctx.i1, 0));
        ctx_write(bldr, rhnorec_ctx, CTX_IS_WRITE_DETECTED,
                  ConstantInt::get(rhnorec_ctx.i1, 0));
        ctx_write(bldr, rhnorec_ctx, CTX_IS_READONLY,
                  ConstantInt::get(rhnorec_ctx.i1, 0));

        // Start slow path.
        bldr.CreateCall(get_slow_path_start(mdl, ctx),
                        ArrayRef<Value*>(rhnorec_ctx.ctx_var));
        bldr.CreateBr(slow_path);

        std::map<BasicBlock *, BasicBlock *> bbmap;
        std::map<Value *, Value *> valmap;
        std::queue<BasicBlock *> bbqueue;

//---
auto make_bb = [&] (const Twine &name)
    {
        return BasicBlock::Create(ctx, name, fcn);
    };

//---
auto get_next_slow = [&] (BasicBlock *next_fast)
    {
        BasicBlock *next_slow = bbmap[next_fast];
        if (nullptr == next_slow) {
            next_slow = make_bb(next_fast->getName() + ".slow");
            bbmap[next_fast] = next_slow;
            bbqueue.push(next_fast);
        }
        return next_slow;
    };

//---
auto get_val = [&] (Value *v)
    {
        Value *ret = valmap[v];
        if (nullptr == ret) ret = v;
        return ret;
    };

//---
auto make_store = [&] (IRBuilder<> &bldr, StoreInst *store)
    {
        Value *val = get_val(store->getOperand(0));
        Value *ptr = get_val(store->getOperand(1));
        Type *val_ty = val->getType();
        unsigned int sz = val_ty->getPrimitiveSizeInBits();

        Type *store_ty = Type::getIntNTy(ctx, sz);
        Value *store_val = bldr.CreateBitCast(val, store_ty);
        Value *store_ptr =
        bldr.CreateBitCast(ptr, PointerType::get(store_ty, 0));

        std::array<Value*, 3> params =
            { rhnorec_ctx.ctx_var, store_ptr, store_val };

        bldr.CreateCall(get_slow_store(mdl, ctx, sz),
                        ArrayRef<Value*>(params));
    };

//---
auto make_load = [&] (IRBuilder<> &bldr, LoadInst *orig)
    {
        Value *ptr = get_val(orig->getOperand(0));
        Type *orig_ptr_ty = ptr->getType();
        Type *orig_ty = dyn_cast<PointerType>(orig_ptr_ty)->getElementType();
        unsigned int sz;
        if (isa<PointerType>(*orig_ty)) {
            sz = 64;
        } else {
            sz = orig_ty->getPrimitiveSizeInBits();
        }
        Type *ptr_ty = PointerType::get(Type::getIntNTy(ctx, sz), 0);
        Value *ptr_cast = bldr.CreateBitCast(ptr, ptr_ty);

        std::array<Value*, 2> params =
            { rhnorec_ctx.ctx_var, ptr_cast };

        Value *load = bldr.CreateCall(get_slow_load(mdl, ctx, sz),
                                      ArrayRef<Value*>(params));
        Value *ret;
        if (isa<PointerType>(*orig_ty)) {
            ret = bldr.CreateIntToPtr(load, orig_ty);
        } else {
            ret = bldr.CreateBitCast(load, orig_ty);
        }
        valmap[orig] = ret;
    };

//---
auto slowify_bb = [&] (BasicBlock *fast_bb)
    {
        BasicBlock *slow_bb = bbmap[fast_bb];
        bool split_me = false;
        IRBuilder<> bldr(ctx);
        bldr.SetInsertPoint(slow_bb);

        for (Instruction &inst : *fast_bb) {

            if (split_me) {
                // This case happens at the end of a transaction.  The end has
                // been found and now we split the fast block and both the fast
                // and slow paths jump into the same code path.
                BasicBlock *succ = SplitBlock(fast_bb, &inst);
                bldr.CreateBr(succ);
                return;
            }

            if (isa<CallInst>(inst)) {
                CallInst *call = dyn_cast<CallInst>(&inst);
                if (call->getCalledFunction()->getName() == transform_end) {
                    // End the transaction.
                    Function *commit = get_slow_path_commit(mdl, ctx);
                    bldr.CreateCall(commit,
                                    ArrayRef<Value*>(rhnorec_ctx.ctx_var));

                    // Split the fast basic block and then jump into the
                    // successor from the slow one.
                    split_me = true;
                } else {
                    std::fprintf(stderr, "FIXME: what to do here?\n");
                    abort();
                }
            } else if (isa<StoreInst>(inst)) {
                StoreInst *store = dyn_cast<StoreInst>(&inst);
                make_store(bldr, store);
            } else if (isa<LoadInst>(inst)) {
                LoadInst *load = dyn_cast<LoadInst>(&inst);
                make_load(bldr, load);
            } else if (isa<BranchInst>(inst)) {
                BranchInst *br = dyn_cast<BranchInst>(&inst);
                if (br->isConditional()) {
                    BasicBlock *then_fast = br->getSuccessor(0);
                    BasicBlock *then_slow = get_next_slow(then_fast);
                    BasicBlock *else_fast = br->getSuccessor(1);
                    BasicBlock *else_slow = get_next_slow(else_fast);
                    Value *cond = get_val(br->getCondition());
                    bldr.CreateCondBr(cond, then_slow, else_slow);
                } else {
                    BasicBlock *next_fast = br->getSuccessor(0);
                    BasicBlock *next_slow = get_next_slow(next_fast);
                    bldr.CreateBr(next_slow);
                }
                // Nothing comes after a branch.
                return;
            } else if (isa<TerminatorInst>(inst)) {
                std::fprintf(stderr, "FIXME: terminator.\n");
                abort();
            } else if (isa<BinaryOperator>(inst)) {
                // Naive Duplicate.
                BinaryOperator *op = dyn_cast<BinaryOperator>(&inst);
                Instruction::BinaryOps code = op->getOpcode();
                Value *left = get_val(op->getOperand(0));
                Value *right = get_val(op->getOperand(1));
                Value *ret = bldr.CreateBinOp(code, left, right);
                valmap[&inst] = ret;
            } else if (isa<GetElementPtrInst>(inst)) {
                // Naive Duplicate.
                GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(&inst);
                Value *ptr = get_val(gep->getPointerOperand());
                std::vector<Value *> idxes;
                for (Value *idx : gep->indices()) {
                    idxes.push_back(get_val(idx));
                }
                Value *ret = bldr.CreateGEP(ptr, ArrayRef<Value*>(idxes));
                valmap[gep] = ret;
            } else {
                std::fprintf(stderr, "unsupported instruction.\n");
                abort();
            }
        }
        std::fprintf(stderr, "Internal error: unreachable.\n");
        abort();
    };

        bbmap[fast_path] = slow_path;
        bbqueue.push(fast_path);

        while (!bbqueue.empty()) {
            // Visit nodes, breadth-first.  Why not depth-first?  PHI nodes
            // could depend on values that haven't been duplicated in a depth-
            // first traversal.
            slowify_bb(bbqueue.front());
            bbqueue.pop();
        }
    }

    void transform_rhnorec (Function &fn)
    {
        Module *mdl = fn.getParent();
        LLVMContext &ctx = fn.getContext();
        BasicBlock &entry = fn.getEntryBlock();

        // FIXME: Identify stack alloca's so that writing into that memory
        // doesn't trigger the second phase.

        rhnorec_ctx_t rhnorec_ctx = prepend_context_struct(mdl, ctx, entry);

        for (BasicBlock &bb : fn) {
            // Iterate through all of the basic blocks in the function.  We
            // don't return early after finding a transaction to transform
            // because the function may have more than one.
            for (Instruction &inst : bb) {
                CallInst *call = dyn_cast<CallInst>(&inst);
                if (!call) continue;
                if (call->getCalledFunction()->getName() == transform_begin) {
                    do_transform(mdl, ctx, rhnorec_ctx, bb, inst);
                }
            }
        }
    }

    // RHNoRec - The first implementation, without getAnalysisUsage.
    struct RHNoRec : public FunctionPass {
        static char ID; // Pass identification, replacement for typeid
        RHNoRec() : FunctionPass(ID) {}

        bool runOnFunction(Function &fn) override {
            ++RHNoRecCounter;
            if (!needs_transform(fn)) return false;
            transform_rhnorec(fn);
            return false;
        }
    };
}

char RHNoRec::ID = 0;
static RegisterPass<RHNoRec> X("rhnorec", "RHNoRec Pass");
