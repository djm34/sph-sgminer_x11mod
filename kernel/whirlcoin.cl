/*
 * whirlcoin kernel implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2014  phm
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
 * @author   djm34
 */
#ifndef W_CL
#define W_CL

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
#define SPH_C64(x)    ((sph_u64)(x ## UL))




#define SPH_T32(x) (as_uint(x))
#define SPH_ROTL32(x, n) rotate(as_uint(x), as_uint(n))
#define SPH_ROTR32(x, n)   SPH_ROTL32(x, (32 - (n)))
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
#define SPH_SMALL_FOOTPRINT_GROESTL 0
#define SPH_GROESTL_BIG_ENDIAN 0
#define SPH_CUBEHASH_UNROLL 0
#define SPH_KECCAK_UNROLL   0
#define SPH_HAMSI_EXPAND_BIG 4


#include "whirlpool.cl"


#define SWAP4(x) as_uint(as_uchar4(x).wzyx)
#define SWAP8(x) as_ulong(as_uchar8(x).s76543210)

#if SPH_BIG_ENDIAN
    #define DEC64E(x) (x)
    #define DEC64BE(x) (*(const __global sph_u64 *) (x));
    #define DEC32LE(x) SWAP4(*(const __global sph_u32 *) (x));
#else
    #define DEC64E(x) SWAP8(x)
    #define DEC64BE(x) SWAP8(*(const __global sph_u64 *) (x));
    #define DEC32LE(x) (*(const __global sph_u32 *) (x));
#endif

#define SHL(x, n)            ((x) << (n))
#define SHR(x, n)            ((x) >> (n))

#define CONST_EXP2    q[i+0] + SPH_ROTL64(q[i+1], 5)  + q[i+2] + SPH_ROTL64(q[i+3], 11) + \
                    q[i+4] + SPH_ROTL64(q[i+5], 27) + q[i+6] + SPH_ROTL64(q[i+7], 32) + \
                    q[i+8] + SPH_ROTL64(q[i+9], 37) + q[i+10] + SPH_ROTL64(q[i+11], 43) + \
                    q[i+12] + SPH_ROTL64(q[i+13], 53) + (SHR(q[i+14],1) ^ q[i+14]) + (SHR(q[i+15],2) ^ q[i+15])


typedef union {
    unsigned char h1[64];
    uint h4[16];
    ulong h8[8];
} hash_t;


__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void whirlpool(__global unsigned char* block, __global hash_t* hashes)
{
   uint gid = get_global_id(0);
    __global hash_t *hash = &(hashes[gid-get_global_offset(0)]);
  

  sph_u64 n0, n1, n2, n3, n4, n5, n6, n7; 
    sph_u64 h0, h1, h2, h3, h4, h5, h6, h7;
    sph_u64 state[8];

    n0 = DEC64LE(block +   0);
    n1 = DEC64LE(block +   8);
    n2 = DEC64LE(block +   16);
    n3 = DEC64LE(block +   24);
    n4 = DEC64LE(block +   32);
    n5 = DEC64LE(block +   40);
    n6 = DEC64LE(block +   48);
    n7 = DEC64LE(block +   56);

    h0 = h1 = h2 = h3 = h4 = h5 = h6 = h7 = 0;

    n0 ^= h0;
    n1 ^= h1;
    n2 ^= h2;
    n3 ^= h3;
    n4 ^= h4;
    n5 ^= h5;
    n6 ^= h6;
    n7 ^= h7;

    for (unsigned r = 0; r < 10; r ++) {
	sph_u64 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

	ROUND_KSCHED(old1_T, h, tmp, old1_RC[r]);
	TRANSFER(h, tmp);
	ROUND_WENC(old1_T, n, h, tmp);
	TRANSFER(n, tmp);
    }

    state[0] = n0 ^ DEC64LE(block +   0);
    state[1] = n1 ^ DEC64LE(block +   8);
    state[2] = n2 ^ DEC64LE(block +   16);
    state[3] = n3 ^ DEC64LE(block +   24);
    state[4] = n4 ^ DEC64LE(block +   32);
    state[5] = n5 ^ DEC64LE(block +   40);
    state[6] = n6 ^ DEC64LE(block +   48);
    state[7] = n7 ^ DEC64LE(block +   56);

    
    n0 = DEC64LE(block +   64);
    n1 = DEC64LE(block +  72);
    n1 &= 0x00000000FFFFFFFF;
    n1 ^= ((sph_u64) gid) << 32;
    n3 = n4 = n5 = n6 = 0;
	n2 = 0x0000000000000080; 
    n7 = 0x8002000000000000;
    sph_u64 temp0,temp1,temp2,temp7;
    temp0 = n0;
    temp1 = n1;
    temp2 = n2;
    temp7 = n7;
    h0 = state[0];
    h1 = state[1];
    h2 = state[2];
    h3 = state[3];
    h4 = state[4];
    h5 = state[5];
    h6 = state[6];
    h7 = state[7];

    n0 ^= h0;
    n1 ^= h1;
    n2 ^= h2;
    n3 ^= h3;
    n4 ^= h4;
    n5 ^= h5;
    n6 ^= h6;
    n7 ^= h7;

    for (unsigned r = 0; r < 10; r ++) {
	sph_u64 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

	ROUND_KSCHED(old1_T, h, tmp, old1_RC[r]);
	TRANSFER(h, tmp);
	ROUND_WENC(old1_T, n, h, tmp);
	TRANSFER(n, tmp);
    }
    
    state[0] ^= n0 ^ temp0;
    state[1] ^= n1 ^ temp1;
    state[2] ^= n2 ^ temp2;
    state[3] ^= n3;
    state[4] ^= n4;
    state[5] ^= n5;
    state[6] ^= n6;
    state[7] ^= n7 ^ temp7;

    for (unsigned i = 0; i < 8; i ++)
	hash->h8[i] = state[i];  


    barrier(CLK_GLOBAL_MEM_FENCE);

}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void whirlpool2(__global hash_t* hashes)
{

    uint gid = get_global_id(0);
    uint offset = get_global_offset(0);
    __global hash_t *hash = &(hashes[gid-offset]);


sph_u64 n0, n1, n2, n3, n4, n5, n6, n7; 
    sph_u64 h0, h1, h2, h3, h4, h5, h6, h7;
    sph_u64 state[8];

    n0 = (hash->h8[0]);
    n1 = (hash->h8[1]);
    n2 = (hash->h8[2]);
    n3 = (hash->h8[3]);
    n4 = (hash->h8[4]);
    n5 = (hash->h8[5]);
    n6 = (hash->h8[6]);
    n7 = (hash->h8[7]);

    h0 = h1 = h2 = h3 = h4 = h5 = h6 = h7 = 0;

    n0 ^= h0;
    n1 ^= h1;
    n2 ^= h2;
    n3 ^= h3;
    n4 ^= h4;
    n5 ^= h5;
    n6 ^= h6;
    n7 ^= h7;

    for (unsigned r = 0; r < 10; r ++) {
	sph_u64 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

	ROUND_KSCHED(old1_T, h, tmp, old1_RC[r]);
	TRANSFER(h, tmp);
	ROUND_WENC(old1_T, n, h, tmp);
	TRANSFER(n, tmp);
    }

    state[0] = n0 ^ (hash->h8[0]);
    state[1] = n1 ^ (hash->h8[1]);
    state[2] = n2 ^ (hash->h8[2]);
    state[3] = n3 ^ (hash->h8[3]);
    state[4] = n4 ^ (hash->h8[4]);
    state[5] = n5 ^ (hash->h8[5]);
    state[6] = n6 ^ (hash->h8[6]);
    state[7] = n7 ^ (hash->h8[7]);

    n0 = 0x80;
    n1 = n2 = n3 = n4 = n5 = n6 = 0;
    n7 = 0x2000000000000;

    h0 = state[0];
    h1 = state[1];
    h2 = state[2];
    h3 = state[3];
    h4 = state[4];
    h5 = state[5];
    h6 = state[6];
    h7 = state[7];

    n0 ^= h0;
    n1 ^= h1;
    n2 ^= h2;
    n3 ^= h3;
    n4 ^= h4;
    n5 ^= h5;
    n6 ^= h6;
    n7 ^= h7;

    for (unsigned r = 0; r < 10; r ++) {
	sph_u64 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

	ROUND_KSCHED(old1_T, h, tmp, old1_RC[r]);
	TRANSFER(h, tmp);
	ROUND_WENC(old1_T, n, h, tmp);
	TRANSFER(n, tmp);
    }

    state[0] ^= n0 ^ 0x80;
    state[1] ^= n1;
    state[2] ^= n2;
    state[3] ^= n3;
    state[4] ^= n4;
    state[5] ^= n5;
    state[6] ^= n6;
    state[7] ^= n7 ^ 0x2000000000000;

    for (unsigned i = 0; i < 8; i ++)
	hash->h8[i] = state[i];
barrier(CLK_GLOBAL_MEM_FENCE);

}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void whirlpool3(__global hash_t* hashes)
{

    uint gid = get_global_id(0);
    uint offset = get_global_offset(0);
    __global hash_t *hash = &(hashes[gid-offset]);


sph_u64 n0, n1, n2, n3, n4, n5, n6, n7; 
    sph_u64 h0, h1, h2, h3, h4, h5, h6, h7;
    sph_u64 state[8];

    n0 = (hash->h8[0]);
    n1 = (hash->h8[1]);
    n2 = (hash->h8[2]);
    n3 = (hash->h8[3]);
    n4 = (hash->h8[4]);
    n5 = (hash->h8[5]);
    n6 = (hash->h8[6]);
    n7 = (hash->h8[7]);

    h0 = h1 = h2 = h3 = h4 = h5 = h6 = h7 = 0;

    n0 ^= h0;
    n1 ^= h1;
    n2 ^= h2;
    n3 ^= h3;
    n4 ^= h4;
    n5 ^= h5;
    n6 ^= h6;
    n7 ^= h7;

    for (unsigned r = 0; r < 10; r ++) {
	sph_u64 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

	ROUND_KSCHED(old1_T, h, tmp, old1_RC[r]);
	TRANSFER(h, tmp);
	ROUND_WENC(old1_T, n, h, tmp);
	TRANSFER(n, tmp);
    }

    state[0] = n0 ^ (hash->h8[0]);
    state[1] = n1 ^ (hash->h8[1]);
    state[2] = n2 ^ (hash->h8[2]);
    state[3] = n3 ^ (hash->h8[3]);
    state[4] = n4 ^ (hash->h8[4]);
    state[5] = n5 ^ (hash->h8[5]);
    state[6] = n6 ^ (hash->h8[6]);
    state[7] = n7 ^ (hash->h8[7]);

    n0 = 0x80;
    n1 = n2 = n3 = n4 = n5 = n6 = 0;
    n7 = 0x2000000000000;

    h0 = state[0];
    h1 = state[1];
    h2 = state[2];
    h3 = state[3];
    h4 = state[4];
    h5 = state[5];
    h6 = state[6];
    h7 = state[7];

    n0 ^= h0;
    n1 ^= h1;
    n2 ^= h2;
    n3 ^= h3;
    n4 ^= h4;
    n5 ^= h5;
    n6 ^= h6;
    n7 ^= h7;

    for (unsigned r = 0; r < 10; r ++) {
	sph_u64 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

	ROUND_KSCHED(old1_T, h, tmp, old1_RC[r]);
	TRANSFER(h, tmp);
	ROUND_WENC(old1_T, n, h, tmp);
	TRANSFER(n, tmp);
    }

    state[0] ^= n0 ^ 0x80;
    state[1] ^= n1;
    state[2] ^= n2;
    state[3] ^= n3;
    state[4] ^= n4;
    state[5] ^= n5;
    state[6] ^= n6;
    state[7] ^= n7 ^ 0x2000000000000;

    for (unsigned i = 0; i < 8; i ++)
	hash->h8[i] = state[i];
barrier(CLK_GLOBAL_MEM_FENCE);
}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void whirlpool4(__global hash_t* hashes, __global uint* output, const ulong target)
{
    uint gid = get_global_id(0);
    uint offset = get_global_offset(0);
    __global hash_t *hash = &(hashes[gid-offset]);


sph_u64 n0, n1, n2, n3, n4, n5, n6, n7; 
    sph_u64 h0, h1, h2, h3, h4, h5, h6, h7;
    sph_u64 state[8];

    n0 = (hash->h8[0]);
    n1 = (hash->h8[1]);
    n2 = (hash->h8[2]);
    n3 = (hash->h8[3]);
    n4 = (hash->h8[4]);
    n5 = (hash->h8[5]);
    n6 = (hash->h8[6]);
    n7 = (hash->h8[7]);

    h0 = h1 = h2 = h3 = h4 = h5 = h6 = h7 = 0;

    n0 ^= h0;
    n1 ^= h1;
    n2 ^= h2;
    n3 ^= h3;
    n4 ^= h4;
    n5 ^= h5;
    n6 ^= h6;
    n7 ^= h7;

    for (unsigned r = 0; r < 10; r ++) {
	sph_u64 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

	ROUND_KSCHED(old1_T, h, tmp, old1_RC[r]);
	TRANSFER(h, tmp);
	ROUND_WENC(old1_T, n, h, tmp);
	TRANSFER(n, tmp);
    }

    state[0] = n0 ^ (hash->h8[0]);
    state[1] = n1 ^ (hash->h8[1]);
    state[2] = n2 ^ (hash->h8[2]);
    state[3] = n3 ^ (hash->h8[3]);
    state[4] = n4 ^ (hash->h8[4]);
    state[5] = n5 ^ (hash->h8[5]);
    state[6] = n6 ^ (hash->h8[6]);
    state[7] = n7 ^ (hash->h8[7]);

    n0 = 0x80;
    n1 = n2 = n3 = n4 = n5 = n6 = 0;
    n7 = 0x2000000000000;

    h0 = state[0];
    h1 = state[1];
    h2 = state[2];
    h3 = state[3];
    h4 = state[4];
    h5 = state[5];
    h6 = state[6];
    h7 = state[7];

    n0 ^= h0;
    n1 ^= h1;
    n2 ^= h2;
    n3 ^= h3;
    n4 ^= h4;
    n5 ^= h5;
    n6 ^= h6;
    n7 ^= h7;

    for (unsigned r = 0; r < 10; r ++) {
	sph_u64 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

	ROUND_KSCHED(old1_T, h, tmp, old1_RC[r]);
	TRANSFER(h, tmp);
	ROUND_WENC(old1_T, n, h, tmp);
	TRANSFER(n, tmp);
    }

    state[0] ^= n0 ^ 0x80;
    state[1] ^= n1;
    state[2] ^= n2;
    state[3] ^= n3;
    state[4] ^= n4;
    state[5] ^= n5;
    state[6] ^= n6;
    state[7] ^= n7 ^ 0x2000000000000;

    for (unsigned i = 0; i < 8; i ++)
	hash->h8[i] = state[i];
barrier(CLK_GLOBAL_MEM_FENCE);


bool result = (hash->h8[3] <= target);
    if (result)
	output[atomic_inc(output+0xFF)] = SWAP4(gid);

}



#endif // W_CL
