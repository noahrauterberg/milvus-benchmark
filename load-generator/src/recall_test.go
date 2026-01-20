package main

import (
	"math"
	"math/rand"
	"testing"
)

func TestEuclideanDistance_IdenticalVectors(t *testing.T) {
	a := []float32{1.0, 2.0, 3.0}
	b := []float32{1.0, 2.0, 3.0}

	dist := euclideanDistance(a, b)

	if dist != 0.0 {
		t.Errorf("Expected distance 0.0 for identical vectors, got %f", dist)
	}
}

func TestEuclideanDistance_SimpleCase(t *testing.T) {
	a := []float32{0.0, 0.0}
	b := []float32{3.0, 4.0}

	dist := euclideanDistance(a, b)

	expected := float32(math.Pow(3, 2) + math.Pow(4, 2))
	if dist != expected {
		t.Errorf("Expected squared distance %f, got %f", expected, dist)
	}
}

func TestEuclideanDistance_NegativeValues(t *testing.T) {
	a := []float32{-1.0, -2.0}
	b := []float32{1.0, 2.0}

	dist := euclideanDistance(a, b)

	expected := float32(math.Pow(-2, 2) + math.Pow(-4, 2))
	if dist != expected {
		t.Errorf("Expected squared distance %f, got %f", expected, dist)
	}
}

func TestEuclideanDistance_HighDimension(t *testing.T) {
	dim := 50
	a := make([]float32, dim)
	b := make([]float32, dim)

	// Set all elements of a to 1.0, b to 0.0 for overall distance of 50 (before sqrt)
	for i := range a {
		a[i] = 1.0
		b[i] = 0.0
	}

	dist := euclideanDistance(a, b)

	expected := float32(50.0)
	if dist != expected {
		t.Errorf("Expected squared distance %f, got %f", expected, dist)
	}
}

func TestSortedNeighborsInsertSorted_InsertIntoEmpty(t *testing.T) {
	h := make(sortedNeighbors, 0)
	n := neighbor{id: 1, distance: 5.0}

	h = h.InsertSorted(n, 3)

	if len(h) != 1 {
		t.Errorf("Expected length 1 after insert into empty, got %d", len(h))
	}
	if h[0].id != 1 || h[0].distance != 5.0 {
		t.Errorf("Expected neighbor {1, 5.0}, got %+v", h[0])
	}
}

func TestSortedNeighborsInsertSorted_InsertSmallerDistance(t *testing.T) {
	h := sortedNeighbors{
		{id: 1, distance: 10.0},
		{id: 2, distance: 20.0},
	}
	n := neighbor{id: 3, distance: 5.0}

	h = h.InsertSorted(n, 3)

	if len(h) != 3 {
		t.Errorf("Expected length 3, got %d", len(h))
	}
	if h[0].id != 3 || h[0].distance != 5.0 {
		t.Errorf("Expected first neighbor to be {3, 5.0}, got %+v", h[0])
	}
}

func TestSortedNeighborsInsertSorted_InsertMiddle(t *testing.T) {
	h := sortedNeighbors{
		{id: 1, distance: 5.0},
		{id: 2, distance: 15.0},
	}
	n := neighbor{id: 3, distance: 10.0}

	h = h.InsertSorted(n, 3)

	if len(h) != 3 {
		t.Errorf("Expected length 3, got %d", len(h))
	}
	if h[1].id != 3 || h[1].distance != 10.0 {
		t.Errorf("Expected middle neighbor to be {3, 10.0}, got %+v", h[1])
	}
}

func TestSortedNeighborsInsertSorted_InsertButDontExceedLengthK(t *testing.T) {
	h := sortedNeighbors{
		{id: 1, distance: 5.0},
		{id: 2, distance: 10.0},
		{id: 3, distance: 15.0},
	}
	n := neighbor{id: 4, distance: 7.0}

	h = h.InsertSorted(n, 3)

	if len(h) != 3 {
		t.Errorf("Expected length 3 (truncated), got %d", len(h))
	}
	if h[0].id != 1 || h[1].id != 4 || h[2].id != 2 {
		t.Errorf("Unexpected order after truncation: %+v", h)
	}
}

func TestSortedNeighborsInsertSorted_DoNotInsertLargerWhenFull(t *testing.T) {
	h := sortedNeighbors{
		{id: 1, distance: 5.0},
		{id: 2, distance: 10.0},
	}
	n := neighbor{id: 3, distance: 15.0}

	h = h.InsertSorted(n, 2)

	if len(h) != 2 {
		t.Errorf("Expected length 2, got %d", len(h))
	}
	for _, neighbor := range h {
		if neighbor.id == 3 {
			t.Error("Neighbor with larger distance should not be inserted when at capacity")
		}
	}
}

func TestSortedNeighborsInsertSorted_AppendWhenLengthLowerK(t *testing.T) {
	h := sortedNeighbors{
		{id: 1, distance: 5.0},
	}
	n := neighbor{id: 2, distance: 10.0}

	h = h.InsertSorted(n, 3)

	if len(h) != 2 {
		t.Errorf("Expected length 2, got %d", len(h))
	}
	if h[1].id != 2 || h[1].distance != 10.0 {
		t.Errorf("Expected second neighbor to be {2, 10.0}, got %+v", h[1])
	}
}

func TestSortedNeighborsInsertSorted_EqualDistance(t *testing.T) {
	h := sortedNeighbors{
		{id: 1, distance: 10.0},
		{id: 2, distance: 20.0},
	}
	n := neighbor{id: 3, distance: 10.0}

	h = h.InsertSorted(n, 3)

	if len(h) != 3 {
		t.Errorf("Expected length 3, got %d", len(h))
	}
	if h[1].id != 3 {
		t.Errorf("Expected id 3 at position 1, got %d", h[1].id)
	}
}

func TestSortedNeighborsInsertSorted_InverseInsertOrder(t *testing.T) {
	h := make(sortedNeighbors, 0)

	h = h.InsertSorted(neighbor{id: 1, distance: 30.0}, 5)
	h = h.InsertSorted(neighbor{id: 2, distance: 20.0}, 5)
	h = h.InsertSorted(neighbor{id: 3, distance: 10.0}, 5)
	h = h.InsertSorted(neighbor{id: 4, distance: 25.0}, 5)
	h = h.InsertSorted(neighbor{id: 5, distance: 5.0}, 5)

	if len(h) != 5 {
		t.Errorf("Expected length 5, got %d", len(h))
	}

	expectedIds := []int64{5, 3, 2, 4, 1}
	for i, expectedId := range expectedIds {
		if h[i].id != expectedId {
			t.Errorf("Expected id %d at position %d, got %d", expectedId, i, h[i].id)
		}
	}
}

func TestNearestNeighbors_BasicCase(t *testing.T) {
	query := Vector{0.0, 0.0}
	rawData := []DataRow{
		{Id: 1, Vector: Vector{1.0, 0.0}},
		{Id: 2, Vector: Vector{2.0, 0.0}},
		{Id: 3, Vector: Vector{0.5, 0.0}},
		{Id: 4, Vector: Vector{3.0, 0.0}},
		{Id: 5, Vector: Vector{0.0, 0.25}},
	}

	result := nearestNeighbors(query, rawData, 3)

	if len(result) != 3 {
		t.Errorf("Expected 3 results, got %d", len(result))
	}

	expected := []int64{5, 3, 1}
	for i, id := range expected {
		if result[i] != id {
			t.Errorf("Expected result[%d] to be %d, got %d", i, id, result[i])
		}
	}
}

func TestNearestNeighbors_SingleNeighbor(t *testing.T) {
	query := Vector{5.0, 5.0}
	rawData := []DataRow{
		{Id: 1, Vector: Vector{0.0, 0.0}},
		{Id: 2, Vector: Vector{4.0, 4.0}},
		{Id: 3, Vector: Vector{10.0, 10.0}},
	}

	result := nearestNeighbors(query, rawData, 1)

	if len(result) != 1 {
		t.Errorf("Expected 1 result, got %d", len(result))
	}
	if result[0] != 2 {
		t.Errorf("Expected nearest neighbor id 2, got %d", result[0])
	}
}

func TestNearestNeighbors_HighDimensional(t *testing.T) {
	dim := 10
	query := make(Vector, dim)
	for i := range query {
		query[i] = 0.0
	}

	rawData := make([]DataRow, 5)
	for i := range rawData {
		vec := make(Vector, dim)
		for j := range vec {
			vec[j] = float32(i + 1) // Distance increases with id
		}
		rawData[i] = DataRow{Id: int64(i + 1), Vector: vec}
	}

	result := nearestNeighbors(query, rawData, 3)

	if len(result) != 3 {
		t.Errorf("Expected 3 results, got %d", len(result))
	}

	expected := []int64{1, 2, 3}
	for i, id := range expected {
		if result[i] != id {
			t.Errorf("Expected result[%d] to be %d, got %d", i, id, result[i])
		}
	}
}

func TestNearestNeighbors_TiedDistances(t *testing.T) {
	query := Vector{0.0, 0.0}
	rawData := []DataRow{
		{Id: 1, Vector: Vector{1.0, 0.0}},
		{Id: 2, Vector: Vector{0.0, 1.0}},
		{Id: 3, Vector: Vector{-1.0, 0.0}},
	}

	result := nearestNeighbors(query, rawData, 2)

	if len(result) != 2 {
		t.Errorf("Expected 2 results, got %d", len(result))
	}

	// With tied distances, the order depends on insertion order
	expected := []int64{1, 2}
	for i, id := range expected {
		if result[i] != id {
			t.Errorf("Expected result[%d] to be %d, got %d", i, id, result[i])
		}
	}
}

func TestNearestNeighbors_ExactMatch(t *testing.T) {
	query := Vector{1.0, 2.0}
	rawData := []DataRow{
		{Id: 1, Vector: Vector{1.0, 2.0}}, // distance = 0
		{Id: 2, Vector: Vector{2.0, 3.0}},
		{Id: 3, Vector: Vector{0.0, 0.0}},
	}

	result := nearestNeighbors(query, rawData, 1)

	if len(result) != 1 {
		t.Errorf("Expected 1 result, got %d", len(result))
	}
	if result[0] != 1 {
		t.Errorf("Expected exact match id 1, got %d", result[0])
	}
}

func TestNearestNeighborsSequential_BasicCase(t *testing.T) {
	query := Vector{0.0, 0.0}
	rawData := []DataRow{
		{Id: 1, Vector: Vector{1.0, 0.0}},
		{Id: 2, Vector: Vector{2.0, 0.0}},
		{Id: 3, Vector: Vector{0.5, 0.0}},
		{Id: 4, Vector: Vector{3.0, 0.0}},
		{Id: 5, Vector: Vector{0.0, 0.25}},
	}

	result := nearestNeighborsSequential(query, rawData, 3)

	if len(result) != 3 {
		t.Errorf("Expected 3 results, got %d", len(result))
	}

	expected := []int64{5, 3, 1}
	for i, id := range expected {
		if result[i].id != id {
			t.Errorf("Expected result[%d].id to be %d, got %d", i, id, result[i].id)
		}
	}
}

func TestMergeNeighbors_BasicMerge(t *testing.T) {
	list1 := sortedNeighbors{
		{id: 1, distance: 1.0},
		{id: 2, distance: 3.0},
	}
	list2 := sortedNeighbors{
		{id: 3, distance: 2.0},
		{id: 4, distance: 4.0},
	}

	merged := mergeNeighbors([]sortedNeighbors{list1, list2}, 3)

	if len(merged) != 3 {
		t.Errorf("Expected 3 results, got %d", len(merged))
	}

	expected := []int64{1, 3, 2}
	for i, id := range expected {
		if merged[i].id != id {
			t.Errorf("Expected merged[%d].id to be %d, got %d", i, id, merged[i].id)
		}
	}
}

func TestMergeNeighbors_EmptyLists(t *testing.T) {
	list1 := sortedNeighbors{}
	list2 := sortedNeighbors{}

	merged := mergeNeighbors([]sortedNeighbors{list1, list2}, 3)

	if len(merged) != 0 {
		t.Errorf("Expected 0 results for empty lists, got %d", len(merged))
	}
}

func TestMergeNeighbors_SingleList(t *testing.T) {
	list := sortedNeighbors{
		{id: 1, distance: 1.0},
		{id: 2, distance: 2.0},
		{id: 3, distance: 3.0},
	}

	merged := mergeNeighbors([]sortedNeighbors{list}, 2)

	if len(merged) != 2 {
		t.Errorf("Expected 2 results, got %d", len(merged))
	}

	expected := []int64{1, 2}
	for i, id := range expected {
		if merged[i].id != id {
			t.Errorf("Expected merged[%d].id to be %d, got %d", i, id, merged[i].id)
		}
	}
}

func TestMergeNeighbors_OverlappingDistances(t *testing.T) {
	list1 := sortedNeighbors{
		{id: 1, distance: 1.0},
		{id: 2, distance: 2.0},
	}
	list2 := sortedNeighbors{
		{id: 3, distance: 1.5},
		{id: 4, distance: 2.5},
	}
	list3 := sortedNeighbors{
		{id: 5, distance: 0.5},
		{id: 6, distance: 3.0},
	}

	merged := mergeNeighbors([]sortedNeighbors{list1, list2, list3}, 4)

	if len(merged) != 4 {
		t.Errorf("Expected 4 results, got %d", len(merged))
	}

	expected := []int64{5, 1, 3, 2}
	for i, id := range expected {
		if merged[i].id != id {
			t.Errorf("Expected merged[%d].id to be %d, got %d", i, id, merged[i].id)
		}
	}
}

func TestNearestNeighbors_LargeDataset(t *testing.T) {
	dim := 10
	dataSize := 10000
	k := 5

	query := make(Vector, dim)
	for i := range query {
		query[i] = 0.0
	}

	// Create data points where distance = id (first point closest)
	rawData := make([]DataRow, dataSize)
	for i := range rawData {
		vec := make(Vector, dim)
		for j := range vec {
			vec[j] = float32(i + 1)
		}
		rawData[i] = DataRow{Id: int64(i + 1), Vector: vec}
	}

	result := nearestNeighbors(query, rawData, k)

	if len(result) != k {
		t.Errorf("Expected %d results, got %d", k, len(result))
	}

	// Closest should be ordered by id
	for i := range k {
		if result[i] != int64(i+1) {
			t.Errorf("Expected result[%d] to be %d, got %d", i, i+1, result[i])
		}
	}
}

func TestNearestNeighbors_ParallelConsistency(t *testing.T) {
	dim := 10
	dataSize := 2000
	k := 10

	query := make(Vector, dim)
	for i := range query {
		query[i] = float32(i)
	}

	rawData := make([]DataRow, dataSize)
	for i := range rawData {
		vec := make(Vector, dim)
		for j := range vec {
			vec[j] = float32(rand.Float32())
		}
		rawData[i] = DataRow{Id: int64(i + 1), Vector: vec}
	}

	result := nearestNeighbors(query, rawData, k)

	// Get result from sequential to verify consistent results
	seqResult := nearestNeighborsSequential(query, rawData, k)

	if len(result) != len(seqResult) {
		t.Errorf("Length mismatch: parallel=%d, sequential=%d", len(result), len(seqResult))
	}

	for i := range result {
		if result[i] != seqResult[i].id {
			t.Errorf("Mismatch at position %d: parallel=%d, sequential=%d",
				i, result[i], seqResult[i].id)
		}
	}
}

func TestCalculateRecall_PerfectRecall(t *testing.T) {
	query := Vector{0.0, 0.0}
	rawData := []DataRow{
		{Id: 1, Vector: Vector{1.0, 0.0}},
		{Id: 2, Vector: Vector{2.0, 0.0}},
		{Id: 3, Vector: Vector{3.0, 0.0}},
	}
	resultIds := []int64{1, 2, 3}

	recall := calculateRecall(query, resultIds, rawData)

	if recall != 1.0 {
		t.Errorf("Expected recall 1.0, got %f", recall)
	}
}

func TestCalculateRecall_ZeroRecall(t *testing.T) {
	query := Vector{0.0, 0.0}
	rawData := []DataRow{
		{Id: 1, Vector: Vector{1.0, 0.0}},
		{Id: 2, Vector: Vector{2.0, 0.0}},
		{Id: 3, Vector: Vector{3.0, 0.0}},
		{Id: 4, Vector: Vector{4.0, 0.0}},
		{Id: 5, Vector: Vector{5.0, 0.0}},
	}
	resultIds := []int64{4, 5}

	recall := calculateRecall(query, resultIds, rawData)

	if recall != 0.0 {
		t.Errorf("Expected recall 0.0, got %f", recall)
	}
}

func TestCalculateRecall_PartialRecall(t *testing.T) {
	query := Vector{0.0, 0.0}
	rawData := []DataRow{
		{Id: 1, Vector: Vector{1.0, 0.0}},
		{Id: 2, Vector: Vector{2.0, 0.0}},
		{Id: 3, Vector: Vector{3.0, 0.0}},
		{Id: 4, Vector: Vector{4.0, 0.0}},
	}
	resultIds := []int64{1, 3}

	recall := calculateRecall(query, resultIds, rawData)

	expected := 0.5
	if math.Abs(recall-expected) > 0.0001 {
		t.Errorf("Expected recall %f, got %f", expected, recall)
	}
}

func TestEnhanceJobResults_SingleJob(t *testing.T) {
	rawData := []DataRow{
		{Id: 1, Vector: Vector{1.0, 0.0}},
		{Id: 2, Vector: Vector{2.0, 0.0}},
		{Id: 3, Vector: Vector{3.0, 0.0}},
	}
	jobs := []Job{
		{
			QueryVector: Vector{0.0, 0.0},
			ResultIds:   []int64{1, 2}, // Perfect recall
		},
	}

	results := EnhanceJobResults(rawData, jobs)

	if len(results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(results))
	}
	if results[0].Recall != 1.0 {
		t.Errorf("Expected recall 1.0, got %f", results[0].Recall)
	}
}

func TestEnhanceJobResults_MultipleJobs(t *testing.T) {
	rawData := []DataRow{
		{Id: 1, Vector: Vector{1.0, 0.0}},
		{Id: 2, Vector: Vector{2.0, 0.0}},
		{Id: 3, Vector: Vector{3.0, 0.0}},
		{Id: 4, Vector: Vector{4.0, 0.0}},
	}
	jobs := []Job{
		{
			QueryVector: Vector{0.0, 0.0},
			ResultIds:   []int64{1, 2}, // Perfect recall
		},
		{
			QueryVector: Vector{0.0, 0.0},
			ResultIds:   []int64{1, 3}, // 50% recall
		},
		{
			QueryVector: Vector{0.0, 0.0},
			ResultIds:   []int64{3, 4}, // 0% recall
		},
	}

	results := EnhanceJobResults(rawData, jobs)

	if len(results) != 3 {
		t.Errorf("Expected 3 results, got %d", len(results))
	}

	expectedRecalls := []float64{1.0, 0.5, 0.0}
	for i, expected := range expectedRecalls {
		if math.Abs(results[i].Recall-expected) > 0.0001 {
			t.Errorf("Expected results[%d].Recall = %f, got %f", i, expected, results[i].Recall)
		}
	}
}

func TestEnhanceJobResults_EmptyJobs(t *testing.T) {
	rawData := []DataRow{
		{Id: 1, Vector: Vector{1.0, 0.0}},
	}
	jobs := []Job{}

	results := EnhanceJobResults(rawData, jobs)

	if len(results) != 0 {
		t.Errorf("Expected 0 results for empty jobs, got %d", len(results))
	}
}

func TestEnhanceJobResults_PreservesJobData(t *testing.T) {
	rawData := []DataRow{
		{Id: 1, Vector: Vector{1.0, 0.0}},
		{Id: 2, Vector: Vector{2.0, 0.0}},
	}
	jobs := []Job{
		{
			QueryVector: Vector{0.0, 0.0},
			ResultIds:   []int64{1},
		},
	}

	results := EnhanceJobResults(rawData, jobs)

	if len(results) != 1 {
		t.Fatalf("Expected 1 result, got %d", len(results))
	}

	if len(results[0].QueryVector) != 2 {
		t.Errorf("QueryVector not preserved")
	}
	if len(results[0].ResultIds) != 1 || results[0].ResultIds[0] != 1 {
		t.Errorf("ResultIds not preserved")
	}
}
