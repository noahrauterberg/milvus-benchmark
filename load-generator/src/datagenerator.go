package main

import (
	"math/rand"
)

func GenerateVector(generator *rand.Rand, dim int, stdDev float32, mean float32) (vector []float32) {
	vector = make([]float32, dim)
	for i := range vector {
		vector[i] = float32(generator.NormFloat64())*stdDev + mean
	}
	return
}

func GenerateQueryVectors(
	generator *rand.Rand,
	dim int,
	numQueries int,
	stdDev float32,
	mean float32,
) [][]float32 {
	vectors := make([][]float32, numQueries)
	for i := range numQueries {
		vectors[i] = GenerateVector(generator, dim, stdDev, mean)
	}
	return vectors
}
