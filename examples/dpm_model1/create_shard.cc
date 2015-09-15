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

float calcAvg(const float *featureMatrix, int patientNum, int windowNum, int featureDim, float *avgVec) {
    int patientWidth = windowNum * featureDim;
    //memset(avgVec, 0, sizeof(float) * patientWidth * patientNum);
    memset(avgVec, 0, sizeof(float) * featureDim);
    for (int k = 0; k < featureDim; ++k) {  // for each feature
        for (int i = 0; i < patientNum; ++i) {  // for each patient
            for (int j = 0; j < windowNum; ++j) {  // for each time window
                avgVec[k] += *(featureMatrix + i * patientWidth + j * featureDim + k);  // aggregate all values for 1 feature
            }
        }
        avgVec[k] /= patientNum * windowNum;  // then compute the average value
    }
}

float calcSD(const float *featureMatrix, int patientNum, int windowNum,
             int featureDim, const float *avgVec, float *sdVec) {
    int patientWidth = windowNum * featureDim;
    //memset(avgVec, 0, sizeof(float) * patientWidth * patientNum);
    memset(sdVec, 0, sizeof(float) * featureDim);
    for (int k = 0; k < featureDim; ++k) {
        for (int i = 0; i < patientNum; ++i) {
            for (int j = 0; j < windowNum; ++j) {
                sdVec[k] += pow(avgVec[k] - (featureMatrix + i * patientWidth + j * featureDim)[k], 2);  // compute variance * N
            }
        }
        sdVec[k] = sqrt(sdVec[k] / (patientNum * windowNum));  // compute standard deviance
    }
}

void normalizeFeature(float *featureMatrix, int patientNum, int windowNum, int featureDim) {
    float *avgVec = new float[featureDim];  // store the average value for one feature
    float *sdVec = new float[featureDim];  // store the standard deviation value for one feature
    calcAvg(featureMatrix, patientNum, windowNum, featureDim, avgVec);
    calcSD(featureMatrix, patientNum, windowNum, featureDim, avgVec, sdVec);
    int patientWidth = windowNum * featureDim;
    for (int i = 0; i < patientNum; ++i) {
        for (int j = 0; j < windowNum; ++j) {
            for (int k = 0; k < featureDim; ++k) {  // both averVec and sdVec are in featureDim dimensions
                float item = (featureMatrix + i * patientWidth + j * featureDim)[k];  // locate this feature value
                if(sdVec[k] > EPS) {
                    item = (item - avgVec[k]) / sdVec[k];  // normalize the this feature value
                }
                else {
                    item = item - avgVec[k];  // if this feature for all patients are "0"
                }
                (featureMatrix + i * patientWidth + j * featureDim)[k] = item;
            }
        }
    }
    delete[] avgVec;
    delete[] sdVec;
}

int generateShardFile(float *featureMatrix, const std::string &filePath, int shardSize, int windowNum,
                      int featureDim, int offset, std::string *nricVec, float *labelVec) {
    DataShard dataShard(filePath, DataShard::kCreate);
    int patientWidth = windowNum * featureDim;

    for (int i = offset; i < offset + shardSize; ++i) {  // for one patient, corresponding to one mvr
        // shardSize here refers to how many patients in train/valid/test shard; offset here can be seen as patient index
        //singa::DPMMultiVectorRecord mvr;
        singa::Record record;
        record.set_type(singa::Record::kDPMMultiVector);
        singa::DPMMultiVectorRecord *mvr = record.mutable_dpm_multi_vector_record();
        for (int j = 0; j < windowNum; ++j) {  // for one time window of one patient, corresponding to one singleVec
            //singa::DPMVectorRecord* singleVec = mvr.add_vectors();
            singa::DPMVectorRecord* singleVec = mvr->add_vectors();
            for (int k = 0; k < featureDim; ++k) {
                singleVec->add_data((featureMatrix + i * patientWidth + j * featureDim)[k]);
            }
        }
        //mvr.set_label(labelVec[i]);
        //dataShard.Insert(nricVec[i].c_str(), mvr);
        mvr->set_label(labelVec[i]);
        dataShard.Insert(nricVec[i].c_str(), record);
    }
    dataShard.Flush();
    return shardSize;
}

void createShard(const char *input, int trainSize, int validSize,
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

    // normalize feature by avg 0 var 1
    normalizeFeature(featureMatrix, patientNum, windowNum, featureDim);

    int offset = 0;
    offset += generateShardFile(featureMatrix, "model1_train_shard", trainSize,
                                windowNum, featureDim, offset, nricVec, labelVec);
    offset += generateShardFile(featureMatrix, "model1_valid_shard", validSize,
                                windowNum, featureDim, offset, nricVec, labelVec);
    offset += generateShardFile(featureMatrix, "model1_test_shard", testSize,
                                windowNum, featureDim, offset, nricVec, labelVec);

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
        createShard(argv[1], trainSize, validSize, testSize);
    }
    return 0;
}
