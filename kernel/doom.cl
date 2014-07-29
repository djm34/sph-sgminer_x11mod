/* Doomcoin kernel implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2014  phm
 * Copyright (c) 2014 Girino Vey
 * Copyright (c) 2014 djm34
 * 
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ===========================(LICENSE END)=============================
 *
 * @author   phm <phm@inbox.com>
 */


#ifndef DOOM_CL
#define DOOM_CL

#if __ENDIAN_LITTLE__
#define SPH_LITTLE_ENDIAN 1
#else
#define SPH_BIG_ENDIAN 1
#endif

#define SPH_UPTR sph_u64
typedef unsigned int sph_u32;


typedef int sph_s32;
#ifndef __OPENCL_VERSION__
typedef unsigned long long sph_u64;
typedef long long sph_s64;
#else
typedef unsigned long sph_u64;
typedef long sph_s64;
#endif


#define SPH_64 1
#define SPH_64_TRUE 1

#define SPH_C32(x)    ((sph_u32)(x ## U))
#define SPH_T32(x) (as_uint(x))
#define SPH_ROTL32(x, n) rotate(as_uint(x), as_uint(n))
#define SPH_ROTR32(x, n)   SPH_ROTL32(x, (32 - (n)))

#define SPH_C64(x)    ((sph_u64)(x ## UL))
#define SPH_T64(x) (as_ulong(x))
#define SPH_ROTL64(x, n) rotate(as_ulong(x), (n) & 0xFFFFFFFFFFFFFFFFUL)
#define SPH_ROTR64(x, n)   SPH_ROTL64(x, (64 - (n)))

#define SPH_ECHO_64 1
#define SPH_KECCAK_64 1
#define SPH_JH_64 1
#define SPH_SIMD_NOCOPY 0
#define SPH_KECCAK_NOCOPY 0
#define SPH_COMPACT_BLAKE_64 0
#define SPH_LUFFA_PARALLEL 0
#ifndef SPH_SMALL_FOOTPRINT_GROESTL
#define SPH_SMALL_FOOTPRINT_GROESTL 0
#endif
#define SPH_GROESTL_BIG_ENDIAN 0

#define SPH_CUBEHASH_UNROLL 0
#define SPH_KECCAK_UNROLL   0

#include "luffa.cl"

#define SWAP4(x) as_uint(as_uchar4(x).wzyx)
#define SWAP8(x) as_ulong(as_uchar8(x).s76543210)

#if SPH_BIG_ENDIAN
    #define DEC64E(x) (x)
    #define DEC64BE(x) (*(const __global sph_u64 *) (x));
    #define DEC32BE(x) (*(const __global sph_u32 *) (x));
#else
    #define DEC64E(x) SWAP8(x)
    #define DEC64BE(x) SWAP8(*(const __global sph_u64 *) (x));
    #define DEC32BE(x) SWAP4(*(const __global sph_u32 *) (x));
#endif

#define SHL(x, n)            ((x) << (n))
#define SHR(x, n)            ((x) >> (n))

#define CONST_EXP2    q[i+0] + SPH_ROTL64(q[i+1], 5)  + q[i+2] + SPH_ROTL64(q[i+3], 11) + \
                    q[i+4] + SPH_ROTL64(q[i+5], 27) + q[i+6] + SPH_ROTL64(q[i+7], 32) + \
                    q[i+8] + SPH_ROTL64(q[i+9], 37) + q[i+10] + SPH_ROTL64(q[i+11], 43) + \
                    q[i+12] + SPH_ROTL64(q[i+13], 53) + (SHR(q[i+14],1) ^ q[i+14]) + (SHR(q[i+15],2) ^ q[i+15])



__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search(__global unsigned char* block, volatile __global uint* output, const ulong target)
{
    
uint gid = get_global_id(0);
    union {
        unsigned char h1[64];
        uint h4[16];
        ulong h8[8];
    } hash;
    // luffa

    sph_u32 V00 = SPH_C32(0x6d251e69), V01 = SPH_C32(0x44b051e0), V02 = SPH_C32(0x4eaa6fb4), V03 = SPH_C32(0xdbf78465), V04 = SPH_C32(0x6e292011), V05 = SPH_C32(0x90152df4), V06 = SPH_C32(0xee058139), V07 = SPH_C32(0xdef610bb);
    sph_u32 V10 = SPH_C32(0xc3b44b95), V11 = SPH_C32(0xd9d2f256), V12 = SPH_C32(0x70eee9a0), V13 = SPH_C32(0xde099fa3), V14 = SPH_C32(0x5d9b0557), V15 = SPH_C32(0x8fc944b3), V16 = SPH_C32(0xcf1ccf0e), V17 = SPH_C32(0x746cd581);
    sph_u32 V20 = SPH_C32(0xf7efc89d), V21 = SPH_C32(0x5dba5781), V22 = SPH_C32(0x04016ce5), V23 = SPH_C32(0xad659c05), V24 = SPH_C32(0x0306194f), V25 = SPH_C32(0x666d1836), V26 = SPH_C32(0x24aa230a), V27 = SPH_C32(0x8b264ae7);
    sph_u32 V30 = SPH_C32(0x858075d5), V31 = SPH_C32(0x36d79cce), V32 = SPH_C32(0xe571f7d7), V33 = SPH_C32(0x204b1f67), V34 = SPH_C32(0x35870c6a), V35 = SPH_C32(0x57e9e923), V36 = SPH_C32(0x14bcb808), V37 = SPH_C32(0x7cde72ce);
    sph_u32 V40 = SPH_C32(0x6c68e9be), V41 = SPH_C32(0x5ec41e22), V42 = SPH_C32(0xc825b7c7), V43 = SPH_C32(0xaffb4363), V44 = SPH_C32(0xf5df3999), V45 = SPH_C32(0x0fc688f1), V46 = SPH_C32(0xb07224cc), V47 = SPH_C32(0x03e86cea);

    DECL_TMP8(M);

    M0 = DEC32BE(block + 0);
    M1 = DEC32BE(block + 4);
    M2 = DEC32BE(block + 8);
    M3 = DEC32BE(block + 12);
    M4 = DEC32BE(block + 16);
    M5 = DEC32BE(block + 20);
    M6 = DEC32BE(block + 24);
    M7 = DEC32BE(block + 28);

    for(uint i = 0; i < 5; i++)
    {
        MI5;
        LUFFA_P5;

        if(i == 0) {
            M0 = DEC32BE(block + 32);
            M1 = DEC32BE(block + 36);
            M2 = DEC32BE(block + 40);
            M3 = DEC32BE(block + 44);
            M4 = DEC32BE(block + 48);
            M5 = DEC32BE(block + 52);
            M6 = DEC32BE(block + 56);
            M7 = DEC32BE(block + 60);
        } else if(i == 1) {
            M0 = DEC32BE(block + 64);
            M1 = DEC32BE(block + 68);
            M2 = DEC32BE(block + 72);
            M3 = SWAP4(gid);
            M4 = 0x80000000;
            M5 = M6 = M7 = 0;
        } else if(i == 2) {
            M0 = M1 = M2 = M3 = M4 = M5 = M6 = M7 = 0;
        } else if(i == 3) {
            hash.h4[1] = V00 ^ V10 ^ V20 ^ V30 ^ V40;
            hash.h4[0] = V01 ^ V11 ^ V21 ^ V31 ^ V41;
            hash.h4[3] = V02 ^ V12 ^ V22 ^ V32 ^ V42;
            hash.h4[2] = V03 ^ V13 ^ V23 ^ V33 ^ V43;
            hash.h4[5] = V04 ^ V14 ^ V24 ^ V34 ^ V44;
            hash.h4[4] = V05 ^ V15 ^ V25 ^ V35 ^ V45;
            hash.h4[7] = V06 ^ V16 ^ V26 ^ V36 ^ V46;
            hash.h4[6] = V07 ^ V17 ^ V27 ^ V37 ^ V47;
        }
    }
    hash.h4[9] = V00 ^ V10 ^ V20 ^ V30 ^ V40;
    hash.h4[8] = V01 ^ V11 ^ V21 ^ V31 ^ V41;
    hash.h4[11] = V02 ^ V12 ^ V22 ^ V32 ^ V42;
    hash.h4[10] = V03 ^ V13 ^ V23 ^ V33 ^ V43;
    hash.h4[13] = V04 ^ V14 ^ V24 ^ V34 ^ V44;
    hash.h4[12] = V05 ^ V15 ^ V25 ^ V35 ^ V45;
    hash.h4[15] = V06 ^ V16 ^ V26 ^ V36 ^ V46;
    hash.h4[14] = V07 ^ V17 ^ V27 ^ V37 ^ V47;	


	barrier(CLK_GLOBAL_MEM_FENCE);
	bool result = (SWAP8(hash.h8[3]) <= target);
    if (result)
        output[atomic_inc(output+0xFF)] = SWAP4(gid);
}

#endif// DOOMCOIN_CL