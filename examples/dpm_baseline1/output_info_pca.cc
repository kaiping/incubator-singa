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
                    const std::string &dataline2, const std::string &dataline3, const std::string &dataline4,
                    const std::string &dataline5, const std::string &dataline6, std::string *nric, float *label) {
    auto OnePatientWin1 = split(dataline1, ' '); // format: 57 features Label NRIC
    auto OnePatientWin2 = split(dataline2, ' '); // format: 63 features Label NRIC
    auto OnePatientWin3 = split(dataline3, ' '); // format: 72 features Label NRIC
    auto OnePatientWin4 = split(dataline4, ' '); // format: 115 features Label NRIC
    auto OnePatientWin5 = split(dataline5, ' '); // format: 155 features Label NRIC
    auto OnePatientWin6 = split(dataline6, ' '); // format: 176 features Label NRIC

    // label segment
    // Now we use hard-coded information: 75 features in win1, 130 features in win2 and 240 features in win3
    *label = atof(OnePatientWin6[176].c_str());  // Label (0-239 are features)
    *nric = OnePatientWin6[176 + 1];  // NRIC is after label info

    // feature segment
    for (int i = 0; i < windowNum * featureDim; ++i) { // the first windowNum * featureDim parts are features
        if(i < 57) {
            float value = atof(OnePatientWin1[i].c_str());
            featureMatrix[i] = value;
        }
        else if(i < 120) {
            float value = atof(OnePatientWin2[i - 57].c_str());
            featureMatrix[i] = value;
        }
        else if(i < 192) {
            float value = atof(OnePatientWin3[i - 120].c_str());
            featureMatrix[i] = value;
        }
        else if(i < 307) {
            float value = atof(OnePatientWin4[i - 192].c_str());
            featureMatrix[i] = value;
        }
        else if(i < 462) {
            float value = atof(OnePatientWin5[i - 307].c_str());
            featureMatrix[i] = value;
        }
        else {
            float value = atof(OnePatientWin6[i - 462].c_str());
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

void printForWEKA(const char *input1, const char *input2, const char *input3, const char *input4, const char *input5, const char *input6) {  // input is the output text file of python

    std::ifstream in1(input1);
    CHECK(in1) << "Unable to open file " << input1;
    std::ifstream in2(input2);
    CHECK(in2) << "Unable to open file " << input2;
    std::ifstream in3(input3);
    CHECK(in3) << "Unable to open file " << input3;
    std::ifstream in4(input4);
    CHECK(in4) << "Unable to open file " << input4;
    std::ifstream in5(input5);
    CHECK(in5) << "Unable to open file " << input5;
    std::ifstream in6(input6);
    CHECK(in6) << "Unable to open file " << input6;


    // Just set number of patient, hard-coded now
    int patientNum = 1818;

    // Just set number of shard for each patient, hard-coded now
    int windowNum = 1;

    // Just set dimension of features, hard-coded now
    int featureDim = 638;

    // malloc mem space for feature matrix [patient][shard][feature_idx]
    float *featureMatrix = new float[patientNum * windowNum * featureDim];
    memset(featureMatrix, 0, sizeof(float) * patientNum * windowNum * featureDim);
    float *labelVec = new float[patientNum];
    std::string *nricVec = new std::string[patientNum];
    std::string dataline1;
    std::string dataline2;
    std::string dataline3;
    std::string dataline4;
    std::string dataline5;
    std::string dataline6;
    //std::cout << "test" << std::endl;
    for (int i = 0; i < patientNum; ++i) {  // Here we control how we use different patients; and here we obtain all info we need
        getline(in1, dataline1);
        getline(in2, dataline2);
        getline(in3, dataline3);
        getline(in4, dataline4);
        getline(in5, dataline5);
        getline(in6, dataline6);
        readOnePatient(featureMatrix + i * windowNum * featureDim, windowNum, featureDim, dataline1, dataline2,
                       dataline3, dataline4, dataline5, dataline6, nricVec + i, labelVec + i); // We use NRIC vector, but NRIC information is output after label
        //std::cout << "Patient " << i << " WIN1 information: " << dataline1 << std::endl;
        //std::cout << "Patient " << i << " WIN2 information: " << dataline2 << std::endl;
        //std::cout << "Patient " << i << " WIN3 information: " << dataline3 << std::endl;
        //std::cout << "Patient " << i << " WIN4 information: " << dataline4 << std::endl;
        //std::cout << "Patient " << i << " WIN5 information: " << dataline5 << std::endl;
        //std::cout << "Patient " << i << " WIN6 information: " << dataline6 << std::endl;
    }

    //printFeatures("output_info.txt", featureMatrix, labelVec, 0, 1800, windowNum, featureDim); // [0,1799], [0,1800)
    //printFeatures("output_info.txt", featureMatrix, labelVec, 0, 1260, windowNum, featureDim); // [0,1259], [0,1260)
    printFeatures("output_info.txt", featureMatrix, labelVec, 1260, 1800, windowNum, featureDim); // [1260,1799], [1260,1800)

    in1.close();
    in2.close();
    in3.close();
    in4.close();
    in5.close();
    in6.close();

    delete[] featureMatrix;
    delete[] nricVec;
    delete[] labelVec;
}

int main(int argc, char **argv) {
    if (argc != 7) {
        std::cout << "Usage:\n"
                "    create_shard_pca.bin [in_text_file1] [in_text_file2] [in_text_file3] [in_text_file4] [in_text_file5] [in_text_file6]\n";
    } else {
        google::InitGoogleLogging(argv[0]);
        printForWEKA(argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]);
    }
    return 0;
}
