#ifndef MARUCOIN_H
#define MARUCOIN_H

#include "miner.h"

extern int x15_test(unsigned char *pdata, const unsigned char *ptarget,
			uint32_t nonce);
extern void x15_regenhash(struct work *work);

#endif /* MARUCOIN_H */
