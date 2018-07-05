/* 

   COMPILER= icc (Intel Compiler for C) To compile it, please run 
   icc -c -march=native -O3 dgemm.c  

*/

#include <immintrin.h>

const char* dgemm_desc = "Improvised dgemm with AVX instructions.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE   64
#define BLOCK_SIZE_M 32
#define BLOCK_SIZE_N 16
#define BLOCK_SIZE_K 32
#endif

#define min(a,b) (((a)<(b))?(a):(b))


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* restrict C)
{



	for(int k_u = 0; k_u < lda; k_u += BLOCK_SIZE)
		for(int i_u = 0; i_u < lda; i_u += BLOCK_SIZE)
			for(int j_u = 0; j_u < lda; j_u += BLOCK_SIZE){


				for (int k_0 = k_u; k_0 < BLOCK_SIZE + k_u; k_0 += BLOCK_SIZE_K)
				{

					for (int i_0 = i_u; i_0 < BLOCK_SIZE + i_u; i_0 += BLOCK_SIZE_M)
					{


						for(int j_0 = j_u; j_0 < BLOCK_SIZE + j_u; j_0 += BLOCK_SIZE_N)
						{



							const int M = min (lda - i_0, BLOCK_SIZE_M);
							const int N = min (lda - j_0, BLOCK_SIZE_N);
							const int K = min (lda - k_0, BLOCK_SIZE_K);




							if(M == BLOCK_SIZE_M){



								for(int j =0; j < N; ++j)
								{

									const int abs_j = j + j_0;


									double* C_s = C + i_0 + abs_j * lda;

									__m256d sum0 =  _mm256_load_pd(C_s);
									__m256d sum1 =  _mm256_load_pd(C_s + 4);
									__m256d sum2 =  _mm256_load_pd(C_s + 8);
									__m256d sum3 =  _mm256_load_pd(C_s + 12);
									__m256d sum4 =  _mm256_load_pd(C_s + 16);
									__m256d sum5 =  _mm256_load_pd(C_s + 20);
									__m256d sum6 =  _mm256_load_pd(C_s + 24);
									__m256d sum7 =  _mm256_load_pd(C_s + 28);





									for(int k = 0; k < K; ++k){

										const int abs_k = k + k_0;	

										double* A_s = A + i_0 + abs_k * lda; 		

										__m256d a_0 = _mm256_load_pd(A_s);
										__m256d a_1 = _mm256_load_pd(A_s + 4);
										__m256d a_2 = _mm256_load_pd(A_s + 8);
										__m256d a_3 = _mm256_load_pd(A_s + 12);
										__m256d a_4 = _mm256_load_pd(A_s + 16);
										__m256d a_5 = _mm256_load_pd(A_s + 20);
										__m256d a_6 = _mm256_load_pd(A_s + 24);
										__m256d a_7 = _mm256_load_pd(A_s + 28);


										const __m256d b = _mm256_set1_pd(*(B + abs_k + abs_j * lda));


										sum0 = _mm256_fmadd_pd(a_0, b, sum0);
										sum1 = _mm256_fmadd_pd(a_1, b, sum1);
										sum2 = _mm256_fmadd_pd(a_2, b, sum2);
										sum3 = _mm256_fmadd_pd(a_3, b, sum3);
										sum4 = _mm256_fmadd_pd(a_4, b, sum4);
										sum5 = _mm256_fmadd_pd(a_5, b, sum5);
										sum6 = _mm256_fmadd_pd(a_6, b, sum6);
										sum7 = _mm256_fmadd_pd(a_7, b, sum7);
									}


									//double* C_s = C + i_0 + abs_j * lda;

									_mm256_store_pd(C_s, sum0) ;
									_mm256_store_pd(C_s + 4, sum1);
									_mm256_store_pd(C_s + 8, sum2);
									_mm256_store_pd(C_s + 12, sum3);
									_mm256_store_pd(C_s + 16, sum4);
									_mm256_store_pd(C_s + 20, sum5);
									_mm256_store_pd(C_s + 24, sum6);
									_mm256_store_pd(C_s + 28, sum7);

								}	







								continue;
							}


							int istart = 0;



							if(M >= (BLOCK_SIZE_M / 2) ){



								for(int j =0; j < N; ++j)
								{

									const int abs_j = j + j_0;

									__m256d sum0 =  _mm256_load_pd(C + i_0 + 0  + abs_j * lda);
									__m256d sum1 =  _mm256_load_pd(C + i_0 + 4  + abs_j * lda);
									__m256d sum2 =  _mm256_load_pd(C + i_0 + 8  + abs_j * lda);
									__m256d sum3 =  _mm256_load_pd(C + i_0 + 12 + abs_j * lda);


									for(int k = 0; k < K; ++k){

										const int abs_k = k + k_0;	


										__m256d a_0 = _mm256_load_pd(A + i_0 + 0  + abs_k * lda);
										__m256d a_1 = _mm256_load_pd(A + i_0 + 4  + abs_k * lda);
										__m256d a_2 = _mm256_load_pd(A + i_0 + 8  + abs_k * lda);
										__m256d a_3 = _mm256_load_pd(A + i_0 + 12 + abs_k * lda);


										const __m256d b = _mm256_set1_pd(*(B + abs_k + abs_j * lda));


										sum0 = _mm256_fmadd_pd(a_0, b, sum0);
										sum1 = _mm256_fmadd_pd(a_1, b, sum1);
										sum2 = _mm256_fmadd_pd(a_2, b, sum2);
										sum3 = _mm256_fmadd_pd(a_3, b, sum3);

									}



									_mm256_store_pd(C + i_0 + 0  + abs_j * lda, sum0);
									_mm256_store_pd(C + i_0 + 4  + abs_j * lda, sum1);
									_mm256_store_pd(C + i_0 + 8  + abs_j * lda, sum2);
									_mm256_store_pd(C + i_0 + 12 + abs_j * lda, sum3);

								}	



								istart += BLOCK_SIZE_M / 2;




							}

							if(M >= istart + (BLOCK_SIZE_M / 4) ){



								for(int j = 0; j < N; ++j)
								{

									const int abs_j = j + j_0;

									__m256d sum0 =  _mm256_load_pd(C + istart + i_0 + 0  + abs_j * lda);
									__m256d sum1 =  _mm256_load_pd(C + istart + i_0 + 4  + abs_j * lda);


									for(int k = 0; k < K; ++k){

										const int abs_k = k + k_0;	


										__m256d a_0 = _mm256_load_pd(A + istart + i_0 + 0  + abs_k * lda);
										__m256d a_1 = _mm256_load_pd(A + istart + i_0 + 4  + abs_k * lda);


										const __m256d b = _mm256_set1_pd(*(B + abs_k + abs_j * lda));


										sum0 = _mm256_fmadd_pd(a_0, b, sum0);
										sum1 = _mm256_fmadd_pd(a_1, b, sum1);

									}



									_mm256_store_pd(C + istart + i_0 + 0  + abs_j * lda, sum0);
									_mm256_store_pd(C + istart + i_0 + 4  + abs_j * lda, sum1);

								}

								istart += BLOCK_SIZE_M / 4;



							}

							if(M >= istart + (BLOCK_SIZE_M / 8) ){



								for(int j = 0; j < N; ++j)
								{

									const int abs_j = j + j_0;

									__m256d sum0 =  _mm256_load_pd(C + istart + i_0 + 0  + abs_j * lda);


									for(int k = 0; k < K; ++k){

										const int abs_k = k + k_0;	


										__m256d a_0 = _mm256_load_pd(A + istart + i_0 + 0  + abs_k * lda);


										const __m256d b = _mm256_set1_pd(*(B + abs_k + abs_j * lda));


										sum0 = _mm256_fmadd_pd(a_0, b, sum0);

									}



									_mm256_store_pd(C + istart + i_0 + 0  + abs_j * lda, sum0);

								}

								istart += BLOCK_SIZE_M / 8;



							}





							for(int j = 0; j < N; ++j)
								for(int i = istart; i < M; ++i)
								{
									double sum = 0;

									const int abs_i = i_0 + i;
									const int abs_j = j_0 + j;	

									for(int k = 0; k < K; ++k)
										sum += A[abs_i + (k_0 + k) * lda] * B[(k_0 + k) + abs_j * lda];

									C[abs_i + abs_j * lda] += sum;	

								}

						}

					}

				}
			}
}
