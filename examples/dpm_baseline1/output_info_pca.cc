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

int printFeatures(const std::string &filePath, float *featureMatrix, float *labelVec,
                  int start_idx, int end_idx, int windowNum, int featureDim) {
    std::ofstream out(filePath);
    CHECK(out) << "unable to open outoput file" << filePath;
    int patientWidth = windowNum * featureDim;
    for(int tmp = 0; tmp < featureDim; tmp++) {
        out << "Feature" << tmp << ",";
    }
    out << "Label\n";

    for (int i = start_idx; i < end_idx; ++i) {
        for (int j = 0; j < windowNum; ++j) {
            for (int k = 0; k < featureDim; ++k) {
                out << (featureMatrix + i * patientWidth + j * featureDim)[k] << ',';
            }
        }
        out << labelVec[i] << std::endl;
    }

    out.close();
}

void printForWEKA(const char *input1, const char *input2, const char *input3) {  // input is the output text file of python

    std::ifstream in1(input1);
    CHECK(in1) << "Unable to open file " << input1;
    std::ifstream in2(input2);
    CHECK(in2) << "Unable to open file " << input2;
    std::ifstream in3(input3);
    CHECK(in3) << "Unable to open file " << input3;


    // Just set number of patient, hard-coded now
    int patientNum = 1818;

    // Just set number of shard for each patient, hard-coded now
    int windowNum = 1;

    // Just set dimension of features, hard-coded now
    int featureDim = 445;

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
    }

    //printFeatures("output_info.txt", featureMatrix, labelVec, 0, 1800, windowNum, featureDim); // [0,1799], [0,1800)
    //printFeatures("output_info.txt", featureMatrix, labelVec, 0, 1260, windowNum, featureDim); // [0,1259], [0,1260)
    printFeatures("output_info.txt", featureMatrix, labelVec, 1260, 1800, windowNum, featureDim); // [1260,1799], [1260,1800)

    in1.close();
    in2.close();
    in3.close();

    delete[] featureMatrix;
    delete[] nricVec;
    delete[] labelVec;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cout << "Usage:\n"
                "    create_shard_pca.bin [in_text_file1] [in_text_file2] [in_text_file3]\n";
    } else {
        google::InitGoogleLogging(argv[0]);
        printForWEKA(argv[1], argv[2], argv[3]);
    }
    return 0;
}
