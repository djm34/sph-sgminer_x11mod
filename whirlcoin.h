#ifndef W_H
#define W_H

#include "miner.h"

extern int W_test(unsigned char *pdata, const unsigned char *ptarget,
			uint32_t nonce);
extern void W_regenhash(struct work *work);

#endif /* W_H */
