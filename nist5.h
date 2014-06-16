#ifndef NIST5_H
#define NIST5_H

#include "miner.h"

extern int nist5_test(unsigned char *pdata, const unsigned char *ptarget,
			uint32_t nonce);
extern void nist5_regenhash(struct work *work);

#endif /* NIST5_H */
