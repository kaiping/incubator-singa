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

void readOnePatient(float *featureMatrix, int windowNum, int featureDim, const std::string &dataline1,
                    const std::string &dataline2, const std::string &dataline3, std::string *nric, float *label) {
    auto OnePatientWin1 = split(dataline1, ' '); // format: 75 features Label NRIC
    auto OnePatientWin2 = split(dataline2, ' '); // format: 130 features Label NRIC
    auto OnePatientWin3 = split(dataline3, ' '); // format: 240 features Label NRIC

    // label segment
    // Now we use hard-coded information: 75 features in win1, 130 features in win2 and 240 features in win3
    *label = atof(OnePatientWin3[240].c_str());  // Label (0-239 are features)
    *nric = OnePatientWin3[240 + 1];  // NRIC is after label info

    // feature segment
    for (int i = 0; i < windowNum * featureDim; ++i) { // the first windowNum * featureDim parts are features
        if(i < 75) {
            float value = atof(OnePatientWin1[i].c_str());
            featureMatrix[i] = value;
        }
        else if(i < 205) {
            float value = atof(OnePatientWin2[i - 75].c_str());
            featureMatrix[i] = value;
        }
        else {
            float value = atof(OnePatientWin3[i - 205].c_str());
            featureMatrix[i] = value;
        }
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
        /*Need to change this part*/
        /*for (int j = 0; j < windowNum; ++j) {  // for one time window of one patient, corresponding to one singleVec
            singa::DPMVectorRecord* singleVec = mvr->add_vectors();
            for (int k = 0; k < featureDim; ++k) {
                singleVec->add_data((featureMatrix + i * patientWidth + j * featureDim)[k]);
                //std::cout << "Patient " << i << " information features: " << k << " " << (featureMatrix + i * patientWidth + j * featureDim)[k] << std::endl;
            }
        }*/
        // 1st window's features (after pca processing): [0,74]
        singa::DPMVectorRecord* singleVecWin1 = mvr->add_vectors();
        for (int k = 0; k < 75; ++k) {
            singleVecWin1->add_data((featureMatrix + i * patientWidth)[k]);
            //std::cout << "Patient " << i << " information features: " << k << " " << (featureMatrix + i * patientWidth)[k] << std::endl;
        }
        // 2nd window's features (after pca processing): [75,204]
        singa::DPMVectorRecord* singleVecWin2 = mvr->add_vectors();
        for (int k = 0; k < 130; ++k) {
            singleVecWin2->add_data((featureMatrix + i * patientWidth)[k + 75]);
            //std::cout << "Patient " << i << " information features: " << k + 75 << " " << (featureMatrix + i * patientWidth)[k + 75] << std::endl;
        }
        // 3rd window's features (after pca processing): [205,444]
        singa::DPMVectorRecord* singleVecWin3 = mvr->add_vectors();
        for (int k = 0; k < 240; ++k) {
            singleVecWin3->add_data((featureMatrix + i * patientWidth)[k + 75 + 130]);
            //std::cout << "Patient " << i << " information features: " << k + 75 + 130 << " " << (featureMatrix + i * patientWidth)[k + 75 + 130] << std::endl;
        }

        mvr->set_label(labelVec[i]);
        //std::cout << "Patient " << i << " information label: " << labelVec[i] << "NRIC" << nricVec[i] << std::endl;
        dataShard.Insert(nricVec[i].c_str(), record);
    }
    dataShard.Flush();
    return shardSize;
}

void createShard(const char *input1, const char *input2, const char *input3, int trainSize, int validSize,
                 int testSize) {  // input is the output text file of python

    std::ifstream in1(input1);
    CHECK(in1) << "Unable to open file " << input1;
    std::ifstream in2(input2);
    CHECK(in2) << "Unable to open file " << input2;
    std::ifstream in3(input3);
    CHECK(in3) << "Unable to open file " << input3;


    // Just set number of patient, hard-coded now
    int patientNum = 1818;

    // Just set number of shard for each patient, hard-coded now; actually 3 windows, but put windowNum = 1 for simplicity
    int windowNum = 1;

    // Just set dimension of features, hard-coded now, meaning the total number of features
    int featureDim = 445;
    int sumSize = trainSize + validSize + testSize;  // this sumSize should be no more than patientNum
    CHECK(sumSize <= patientNum) << "no enough patients for generation";

    // malloc mem space for feature matrix [patient][shard][feature_idx]
    float *featureMatrix = new float[patientNum * windowNum * featureDim];
    memset(featureMatrix, 0, sizeof(float) * patientNum * windowNum * featureDim);
    float *labelVec = new float[patientNum];
    std::string *nricVec = new std::string[patientNum];
    std::string dataline1;
    std::string dataline2;
    std::string dataline3;
    //std::cout << "test" << std::endl;
    for (int i = 0; i < patientNum; ++i) {  // Here we control how we use different patients; and here we obtain all info we need
        getline(in1, dataline1);
        getline(in2, dataline2);
        getline(in3, dataline3);
        readOnePatient(featureMatrix + i * windowNum * featureDim, windowNum, featureDim, dataline1, dataline2,
                       dataline3, nricVec + i, labelVec + i); // We use NRIC vector, but NRIC information is output after label
        //std::cout << "Patient " << i << " WIN1 information: " << dataline1 << std::endl;
        //std::cout << "Patient " << i << " WIN2 information: " << dataline2 << std::endl;
        //std::cout << "Patient " << i << " WIN3 information: " << dataline3 << std::endl;
    }

    int offset = 0;
    offset += generateShardFile(featureMatrix, "dpm_model1_train_shard", trainSize,
                                windowNum, featureDim, offset, nricVec, labelVec);
    offset += generateShardFile(featureMatrix, "dpm_model1_valid_shard", validSize,
                                windowNum, featureDim, offset, nricVec, labelVec);
    offset += generateShardFile(featureMatrix, "dpm_model1_test_shard", testSize,
                                windowNum, featureDim, offset, nricVec, labelVec);

    in1.close();
    in2.close();
    in3.close();

    delete[] featureMatrix;
    delete[] nricVec;
    delete[] labelVec;
}

int main(int argc, char **argv) {
    if (argc != 7) {
        std::cout << "Usage:\n"
                "    create_shard_pca.bin [in_text_file1] [in_text_file2] [in_text_file3] [train_size] [validate_size] [test_size]\n";
    } else {
        google::InitGoogleLogging(argv[0]);
        int trainSize = atoi(argv[4]);
        int validSize = atoi(argv[5]);
        int testSize = atoi(argv[6]);
        createShard(argv[1], argv[2], argv[3], trainSize, validSize, testSize);
    }
    return 0;
}
