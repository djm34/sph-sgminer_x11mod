/*-
 * Copyright 2009 Colin Percival, 2014 phm
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * This file was originally written by Colin Percival as part of the Tarsnap
 * online backup system.
 */

#include "config.h"
#include "miner.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>



#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"
#include "sph/sph_whirlpool.h"

/* Move init out of loop, so init once externally, and then use one single memcpy with that bigger memory block */
typedef struct {
   /* 
    sph_shavite512_context  shavite1,shavite2;
    sph_simd512_context     simd1,simd2;
    sph_echo512_context     echo1;
	*/
sph_whirlpool1_context    whirlpool1,whirlpool2,whirlpool3,whirlpool4;
} Fhash_context_holder;

Fhash_context_holder base_contexts;


void init_Fhash_contexts()
{
    /*
    sph_shavite512_init(&base_contexts.shavite1);
    sph_simd512_init(&base_contexts.simd1);
	sph_shavite512_init(&base_contexts.shavite2);
	sph_simd512_init(&base_contexts.simd2);
    sph_echo512_init(&base_contexts.echo1);
	*/
	sph_whirlpool1_init(&base_contexts.whirlpool1);
	sph_whirlpool1_init(&base_contexts.whirlpool2);
	sph_whirlpool1_init(&base_contexts.whirlpool3);
	sph_whirlpool1_init(&base_contexts.whirlpool4);
}


/*
 * Encode a length len/4 vector of (uint32_t) into a length len vector of
 * (unsigned char) in big-endian form.  Assumes len is a multiple of 4.
 */
static inline void
be32enc_vect(uint32_t *dst, const uint32_t *src, uint32_t len)
{
	uint32_t i;

	for (i = 0; i < len; i++)
		dst[i] = htobe32(src[i]);
}


inline void fhash(void *state, const void *input)
{
    init_Fhash_contexts();
    
    Fhash_context_holder ctx;
    
    uint32_t hashA[16], hashB[16];  
    //luffa-cubehash-shivite-simd-echo
    memcpy(&ctx, &base_contexts, sizeof(base_contexts));
    /*
    sph_shavite512 (&ctx.shavite1, input, 80);
    sph_shavite512_close (&ctx.shavite1, hashA);    
        
    sph_simd512 (&ctx.simd1, hashA, 64);   
    sph_simd512_close(&ctx.simd1, hashB); 
    
    sph_shavite512 (&ctx.shavite2, hashB, 64);   
    sph_shavite512_close(&ctx.shavite2, hashA);  
    
    sph_simd512 (&ctx.simd2, hashA, 64);   
    sph_simd512_close(&ctx.simd2, hashB); 
    
    sph_echo512 (&ctx.echo1, hashB, 64);   
    sph_echo512_close(&ctx.echo1, hashA);    
	*/
	
    sph_whirlpool1 (&ctx.whirlpool1, input, 80);
    sph_whirlpool1_close (&ctx.whirlpool1, hashA);      
    
	sph_whirlpool1(&ctx.whirlpool2, hashA, 64);
    sph_whirlpool1_close(&ctx.whirlpool2, hashB);

    sph_whirlpool1(&ctx.whirlpool3, hashB, 64);
    sph_whirlpool1_close(&ctx.whirlpool3, hashA);
	
	sph_whirlpool1(&ctx.whirlpool4, hashA, 64);
    sph_whirlpool1_close(&ctx.whirlpool4, hashB);

    memcpy(state, hashB, 32);
}

static const uint32_t diff1targ = 0x0000ffff;


/* Used externally as confirmation of correct OCL code */
int fresh_test(unsigned char *pdata, const unsigned char *ptarget, uint32_t nonce)
{
	uint32_t tmp_hash7, Htarg = le32toh(((const uint32_t *)ptarget)[7]);
	uint32_t data[20], ohash[8];
	//char *scratchbuf;

	be32enc_vect(data, (const uint32_t *)pdata, 19);
	data[19] = htobe32(nonce);
	//scratchbuf = alloca(SCRATCHBUF_SIZE);
	fhash(ohash, data);
	tmp_hash7 = be32toh(ohash[7]);

	applog(LOG_DEBUG, "htarget %08lx diff1 %08lx hash %08lx",
				(long unsigned int)Htarg,
				(long unsigned int)diff1targ,
				(long unsigned int)tmp_hash7);
	if (tmp_hash7 > diff1targ)
		return -1;
	if (tmp_hash7 > Htarg)
		return 0;
	return 1;
}

void fresh_regenhash(struct work *work)
{
        uint32_t data[20];
        char *scratchbuf;
        uint32_t *nonce = (uint32_t *)(work->data + 76);
        uint32_t *ohash = (uint32_t *)(work->hash);

        be32enc_vect(data, (const uint32_t *)work->data, 19);
        data[19] = htobe32(*nonce);
        fhash(ohash, data);
}

bool scanhash_fresh(struct thr_info *thr, const unsigned char __maybe_unused *pmidstate,
		     unsigned char *pdata, unsigned char __maybe_unused *phash1,
		     unsigned char __maybe_unused *phash, const unsigned char *ptarget,
		     uint32_t max_nonce, uint32_t *last_nonce, uint32_t n)
{
	uint32_t *nonce = (uint32_t *)(pdata + 76);
	char *scratchbuf;
	uint32_t data[20];
	uint32_t tmp_hash7;
	uint32_t Htarg = le32toh(((const uint32_t *)ptarget)[7]);
	bool ret = false;

	be32enc_vect(data, (const uint32_t *)pdata, 19);

	while(1) {
		uint32_t ostate[8];

		*nonce = ++n;
		data[19] = (n);
		fhash(ostate, data);
		tmp_hash7 = (ostate[7]);

		applog(LOG_INFO, "data7 %08lx",
					(long unsigned int)data[7]);

		if (unlikely(tmp_hash7 <= Htarg)) {
			((uint32_t *)pdata)[19] = htobe32(n);
			*last_nonce = n;
			ret = true;
			break;
		}

		if (unlikely((n >= max_nonce) || thr->work_restart)) {
			*last_nonce = n;
			break;
		}
	}

	return ret;
}



