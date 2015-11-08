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

void readOnePatient(float *featureMatrix, int windowNum, int featureDim,
                    const std::string &dataLine, std::string *nric, float *label) {
    auto OnePatient = split(dataLine, ' ');
    // NRIC segment
    *nric = OnePatient[0];  // 1st part is NRIC information

    // feature segment
    for (int i = 1; i < windowNum * featureDim + 1; ++i) {
        float value = atof(OnePatient[i].c_str());
        featureMatrix[i] = value;
    }
    // label segment
    *label = atof(OnePatient[windowNum * featureDim + 1].c_str());  // last part is label information
}

int generateShardFile(float *featureMatrix, const std::string &filePath, int shardSize, int windowNum,
                      int featureDim, int offset, std::string *nricVec, float *labelVec) {
    DataShard dataShard(filePath, DataShard::kCreate);
    int patientWidth = windowNum * featureDim;

    for (int i = offset; i < offset + shardSize; ++i) {  // for one patient, corresponding to one mvr
        // shardSize here refers to how many patients in train/valid/test shard; offset here can be seen as patient index
        singa::Record record;
        record.set_type(singa::Record::kDPMMultiVector);
        singa::DPMMultiVectorRecord *mvr = record.mutable_dpm_multi_vector_record();
        for (int j = 0; j < windowNum; ++j) {  // for one time window of one patient, corresponding to one singleVec
            singa::DPMVectorRecord* singleVec = mvr->add_vectors();
            for (int k = 0; k < featureDim; ++k) {
                singleVec->add_data((featureMatrix + i * patientWidth + j * featureDim)[k]);
            }
        }
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

    // Just set number of patient, hard-coded now
    int patientNum = 1641;

    // Just set number of shard for each patient, hard-coded now
    int windowNum = 1;

    // Just set dimension of features, hard-coded now
    int featureDim = 3;

    int sumSize = trainSize + validSize + testSize;  // this sumSize should be no more than patientNum
    CHECK(sumSize <= patientNum) << "no enough patients for generation";

    // malloc mem space for feature matrix [patient][shard][feature_idx]
    float *featureMatrix = new float[patientNum * windowNum * featureDim];
    memset(featureMatrix, 0, sizeof(float) * patientNum * windowNum * featureDim);
    float *labelVec = new float[patientNum];
    std::string *nricVec = new std::string[patientNum];
    std::string dataLine;

    for (int i = 0; i < patientNum; ++i) {
        getline(in, dataLine);
        readOnePatient(featureMatrix + i * windowNum * featureDim, windowNum, featureDim, dataLine,
                       nricVec + i, labelVec + i);
    }

    int offset = 0;
    offset += generateShardFile(featureMatrix, "dpm_baseline1_train_shard_win3", trainSize,
                                windowNum, featureDim, offset, nricVec, labelVec);
    offset += generateShardFile(featureMatrix, "dpm_baseline1_valid_shard_win3", validSize,
                                windowNum, featureDim, offset, nricVec, labelVec);
    offset += generateShardFile(featureMatrix, "dpm_baseline1_test_shard_win3", testSize,
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
