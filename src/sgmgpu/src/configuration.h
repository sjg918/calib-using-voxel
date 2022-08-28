/**
    This file is part of sgm. (https://github.com/dhernandez0/sgm).

    Copyright (c) 2016 Daniel Hernandez Juarez.

    sgm is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    sgm is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with sgm.  If not, see <http://www.gnu.org/licenses/>.

**/

#ifndef CONFIGURATION_H_
#define CONFIGURATION_H_

#include <stdint.h>
#include <stdio.h>
#include <iostream>

//#define PATH_AGGREGATION	4
#define	MAX_DISPARITY		128
#define CENSUS_WIDTH		9
#define CENSUS_HEIGHT		7

#define OCCLUDED_PIXEL		128
#define MISMATCHED_PIXEL	129

#define TOP				(CENSUS_HEIGHT-1)/2
#define LEFT			(CENSUS_WIDTH-1)/2

#define GPU_THREADS_PER_BLOCK_FERMI 256
#define GPU_THREADS_PER_BLOCK_MAXWELL 64

#define FERMI      False
#if FERMI
  #define GPU_THREADS_PER_BLOCK   GPU_THREADS_PER_BLOCK_FERMI
#else
  #define GPU_THREADS_PER_BLOCK   GPU_THREADS_PER_BLOCK_MAXWELL
#endif

#define MAX_COST		30

#define BLOCK_SIZE					256
#define COSTAGG_BLOCKSIZE			GPU_THREADS_PER_BLOCK
#define COSTAGG_BLOCKSIZE_HORIZ		GPU_THREADS_PER_BLOCK

#define ABS_THRESH 3.0
#define REL_THRESH 0.05
#define WARP_SIZE  32

#define ITER_COPY			0
#define ITER_NORMAL			1

#define MIN_COMPUTE			0
#define MIN_NOCOMPUTE		1

#define DIR_UPDOWN			0
#define DIR_DOWNUP			1
#define DIR_LEFTRIGHT		2
#define DIR_RIGHTLEFT		3

#endif /* CONFIGURATION_H_ */
