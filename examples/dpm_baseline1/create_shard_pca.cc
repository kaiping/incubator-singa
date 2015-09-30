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

void readOnePatient(float *featureMatrix, int windowNum, int featureDim,
                    const std::string &dataLine, std::string *nric, float *label) {
    auto OnePatient = split(dataLine, ' '); // format: features (win_num * fea_dim) Label NRIC
    // label segment
    *label = atof(OnePatient[windowNum * featureDim].c_str());  // last part in lavelA
    *nric = OnePatient[windowNum * featureDim + 1];  // NRIC is after label info

    // feature segment
    for (int i = 0; i < windowNum * featureDim; ++i) { // the first windowNum * featureDim parts are features
        float value = atof(OnePatient[i].c_str());
        featureMatrix[i] = value;
    }
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
            singa::DPMVectorRecord* singleVec = mvr->add_vectors();
            for (int k = 0; k < featureDim; ++k) {
                singleVec->add_data((featureMatrix + i * patientWidth + j * featureDim)[k]);
                //std::cout << "Patient " << i << " information features: " << (featureMatrix + i * patientWidth + j * featureDim)[k] << std::endl;
            }
        }
        mvr->set_label(labelVec[i]);
        //std::cout << "Patient " << i << " information label: " << labelVec[i] << std::endl;
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
    int patientNum = 1818;

    // Just set number of shard for each patient, hard-coded now
    int windowNum = 1;

    // Just set dimension of features, hard-coded now
    int featureDim = 280;
    int sumSize = trainSize + validSize + testSize;  // this sumSize should be no more than patientNum
    CHECK(sumSize <= patientNum) << "no enough patients for generation";

    // malloc mem space for feature matrix [patient][shard][feature_idx]
    float *featureMatrix = new float[patientNum * windowNum * featureDim];
    memset(featureMatrix, 0, sizeof(float) * patientNum * windowNum * featureDim);
    float *labelVec = new float[patientNum];
    std::string *nricVec = new std::string[patientNum];
    std::string dataLine;
    //std::cout << "test" << std::endl;
    for (int i = 0; i < patientNum; ++i) {  // Here we control how we use different patients; and here we obtain all info we need
        getline(in, dataLine);
        readOnePatient(featureMatrix + i * windowNum * featureDim, windowNum, featureDim, dataLine,
                       nricVec + i, labelVec + i); // We use NRIC vector, but NRIC information is output after label
        //std::cout << "Patient " << i << " information: " << dataLine << std::endl;
        getline(in, dataLine); // there is an extra blank line after each data line now.
    }

    int offset = 0;
    offset += generateShardFile(featureMatrix, "dpm_baseline1_pca_train_shard", trainSize,
                                windowNum, featureDim, offset, nricVec, labelVec);
    offset += generateShardFile(featureMatrix, "dpm_baseline1_pca_valid_shard", validSize,
                                windowNum, featureDim, offset, nricVec, labelVec);
    offset += generateShardFile(featureMatrix, "dpm_baseline1_pca_test_shard", testSize,
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
