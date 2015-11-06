#include <gflags/gflags.h>
#include <glog/logging.h>
#include <fstream>
#include <string>
#include <iostream>
#include <map>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

#include "utils/data_shard.h"
#include "utils/common.h"
#include "proto/common.pb.h"
//#include "proto/user.pb.h"


using singa::DataShard;

std::vector <std::string> split(const std::string &s, char delim) {
    std::vector <std::string> elems;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.emplace_back(item);
    }
    return elems;
}

const char ctrlA = '\001';
const char ctrlB = '\002';
const char ctrlC = '\003';
const char ctrlD = '\004';
const float EPS = 1e-10;


void readOneWindow(float *featureMatrix, const std::string &shardStr) {
    auto items = split(shardStr, ctrlC);
    for (int i = 0; i != items.size(); ++i) {
        if ("" == items[i]) continue;
        auto tuple = split(items[i], ctrlD);
        CHECK(2 == tuple.size()) << "parse vector data failed.";  // add enough "CHECK"
        int idx = atoi(tuple[0].c_str());
        float value = atof(tuple[1].c_str());
        featureMatrix[idx] = value;
    }
}

void readOnePatient(float *featureMatrix, int windowNum, int featureDim,
                    const std::string &dataLine, std::string *nric, float *label) {
    auto levelA = split(dataLine, ctrlA);
    CHECK(windowNum + 2 == levelA.size()) << "parse line data failed.";  // the 2: one is NRIC, the other is label

    // feature segment
    for (int i = 0; i < windowNum; ++i) {
        readOneWindow(featureMatrix + i * featureDim,
                      levelA[i + 1]);  // the change of "1" is because the 1st string is NRIC
    }

    // label segment
    *label = atof(levelA[windowNum + 1].c_str());  // last part in lavelA
    *nric = levelA[0];  // 1st part in lavelA
}

void output_for_1_win(float *featureMatrix, int trainSize, int windowIdx, int windowNum, int featureDim,
                      std::string *nricVec, float *labelVec, const std::string &filePath) {
    std::ofstream out(filePath);
    int patientWidth = windowNum * featureDim;
    //For training data set, remove empty windows
    for(int i = 0; i < trainSize; i++) {
        int sumFeatureCnt = 0; // for 1 patient, check whether this certain window is empty
        for(int k = 0; k < featureDim; k++) {
            sumFeatureCnt += (featureMatrix + i * patientWidth + windowIdx * featureDim)[k];
            // out << (featureMatrix + i * patientWidth + windowIdx * featureDim)[k] << " ";
        }
        if(sumFeatureCnt > EPS) { // only output non-empty feature windows
            out << nricVec[i] <<" " << "CNT_SUM: " << sumFeatureCnt << std::endl;
            for(int k = 0; k < featureDim; k++) {
                out << (featureMatrix + i * patientWidth + windowIdx * featureDim)[k] << " ";
            }
            out << labelVec[i] << std::endl;
        }
    }
    //For testing dataset, no need to remove empty windows; so no need to check
    for(int i = trainSize; i < nricVec->size(); i++) {
        out << nricVec[i] <<" ";
        for(int k = 0; k < featureDim; k++) {
            out << (featureMatrix + i * patientWidth + windowIdx * featureDim)[k] << " ";
        }
        out << labelVec[i] << std::endl;
    }
}

void output_info(const char *input, int trainSize, int validSize,
                 int testSize) {  // input is the output text file of python

    std::ifstream in(input);
    CHECK(in) << "Unable to open file " << input;

    // read number of patient
    int patientNum;
    in >> patientNum;
    // read number of shard for each patient
    int windowNum;
    in >> windowNum;

    // read dimension of features
    int featureDim;
    in >> featureDim;
    int sumSize = trainSize + validSize + testSize;  // this sumSize should be no more than patientNum
    CHECK(sumSize <= patientNum) << "no enough patients for generation";

    // malloc mem space for feature matrix [patient][shard][feature_idx]
    float *featureMatrix = new float[patientNum * windowNum * featureDim];
    memset(featureMatrix, 0, sizeof(float) * patientNum * windowNum * featureDim);
    float *labelVec = new float[patientNum];
    std::string *nricVec = new std::string[patientNum];
    std::string dataLine;
    getline(in, dataLine);
    for (int i = 0; i < patientNum; ++i) {
        getline(in, dataLine);
        readOnePatient(featureMatrix + i * windowNum * featureDim, windowNum, featureDim, dataLine,
                       nricVec + i, labelVec + i);
    }

    //Now have access to information of all patients in all feature windows
    //Only need to remove empty windows for training data
    //Do not need to remove for valid data and test data

    //Output information window by window
    output_for_1_win(featureMatrix, trainSize, 0, windowNum, featureDim, nricVec, labelVec, "output_win1.txt");
    output_for_1_win(featureMatrix, trainSize, 1, windowNum, featureDim, nricVec, labelVec, "output_win2.txt");
    output_for_1_win(featureMatrix, trainSize, 2, windowNum, featureDim, nricVec, labelVec, "output_win3.txt");

    in.close();

    delete[] featureMatrix;
    delete[] nricVec;
    delete[] labelVec;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        std::cout << "Usage:\n"
                "    create_shard.bin [in_text_file] [train_size] [validate_size] [test_size]\n";
    } else {
        google::InitGoogleLogging(argv[0]);
        int trainSize = atoi(argv[2]);
        int validSize = atoi(argv[3]);
        int testSize = atoi(argv[4]);
        output_info(argv[1], trainSize, validSize, testSize);
    }
    return 0;
}
