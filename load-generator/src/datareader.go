package main

import (
	"bufio"
	"encoding/gob"
	"os"
	"strconv"
	"strings"
)

type Vector []float32

type DataRow struct {
	Id     int64
	Vector Vector
	Word   string
}

type DataSource interface {
	GetDataSet() ([]DataRow, error)
	ReadDataRows() ([]DataRow, error)
}

type DataReader struct {
	sourceFile string
}

// Note: Not used currently
type DataGenerator struct {
	size   int // number of vectors to generate
	mean   float64
	stdDev float64
	dim    int
}

func parseVector(vector []string) Vector {
	ret := make([]float32, len(vector))
	for idx, num := range vector {
		parsedNum, err := strconv.ParseFloat(num, 32)
		if err != nil {
			panic(err)
		}
		ret[idx] = float32(parsedNum)
	}
	return ret
}

func (r DataReader) GetDataSet() ([]DataRow, error) {
	file, err := os.Open(r.sourceFile)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	var rows []DataRow
	for id := int64(0); scanner.Scan(); id++ {
		line := scanner.Text()
		// skip empty lines
		if strings.TrimSpace(line) == "" {
			continue
		}
		parts := strings.Split(line, " ")
		rows = append(rows, DataRow{Id: id, Word: parts[0], Vector: parseVector(parts[1:])})
	}

	return rows, nil
}

func (r DataReader) ReadDataRows() ([]DataRow, error) {
	gobFile, err := os.Open(outputPath("data-rows.gob"))
	if err != nil {
		return nil, err
	}

	var data []DataRow
	decoder := gob.NewDecoder(gobFile)
	err = decoder.Decode(&data)
	return data, err
}
