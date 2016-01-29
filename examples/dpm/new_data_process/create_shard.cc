#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#include "singa/io/store.h"
#include "singa/utils/common.h"
#include "singa/proto/common.pb.h"
#include "./dpm.pb.h"

#define MAX_STRING 100
#define BUFFER_LEN 32
char key[BUFFER_LEN];

char train_file[MAX_STRING];
char test_file[MAX_STRING];

int test_mode;

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
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


void insert_feature_shard(const std::string &feature_string, singa::io::Store *store, int *reccnt) {
    DynamicRecord dynamicRecord;

    auto records = split(feature_string, ctrlB);
    for (auto it = records.begin(); it != records.end(); it++) {
        auto items = split(*it, ctrlC);
        int pid = atoi(items[0].c_str());
        int lt = atoi(items[1].c_str());
        float age = atof(items[4].c_str());
        float edu = atof(items[5].c_str());
        float gen = atof(items[6].c_str());
        int scnt = atoi(items[7].c_str());

        dynamicRecord.set_patient_id(pid);
        dynamicRecord.set_lap_time(lt);
        dynamicRecord.set_age(age);
        dynamicRecord.set_education(edu);
        dynamicRecord.set_gender(gen);
        dynamicRecord.set_nb_sample(scnt);

        auto f_idx = split(items[2], ctrlD);
        auto f_val = split(items[3], ctrlD);
        for (int k = 0; k < f_idx.size(); k++) {
            dynamicRecord.add_observed_idx(atoi(f_idx[k].c_str()));
            dynamicRecord.add_feature_value(atof(f_val[k].c_str()));
        }

        int length = snprintf(key, BUFFER_LEN, "%05d", (*reccnt)++);
        std::string value;
        dynamicRecord.SerializeToString(&value);
        store->Write(std::string(key, length), value);

        dynamicRecord.Clear();
    }

    // append -1 separator
    dynamicRecord.set_patient_id(-1);
    int length = snprintf(key, BUFFER_LEN, "%05d", (*reccnt)++);
    std::string value;
    dynamicRecord.SerializeToString(&value);
    store->Write(std::string(key, length), value);
}

void insert_label_shard(const std::string &label_string, singa::io::Store *store, int *otimecnt) {
    OutTimeRecord outtimeRecord;

    auto items = split(label_string, ctrlB);
    int pid = atoi(items[0].c_str());
    int delta_time = atoi(items[1].c_str());
    float fval = atof(items[2].c_str());

    outtimeRecord.set_patient_id(pid);
    outtimeRecord.set_delta_time(delta_time);
    outtimeRecord.set_mmscore(fval);

    int length = snprintf(key, BUFFER_LEN, "%05d", (*otimecnt)++);
    std::string value;
    outtimeRecord.SerializeToString(&value);
    store->Write(std::string(key, length), value);
}

int create_data(const char *input_file, const char *feature_file, const char *label_file) {

    std::ifstream in(input_file);

    auto *feature_store = singa::io::OpenStore("kvfile", feature_file, singa::io::kCreate);
    auto *label_store = singa::io::OpenStore("kvfile", label_file, singa::io::kCreate);

    int reccnt = 0, otimecnt = 0;
    std::string line;
    while (std::getline(in, line)) {
        auto levelA = split(line, ctrlA);
        insert_feature_shard(levelA[0], feature_store, &reccnt);
        insert_label_shard(levelA[1], label_store, &otimecnt);
    }

    in.close();
    feature_store->Flush();
    label_store->Flush();
    delete feature_store;
    delete label_store;
    return 0;
}


int argPos(char *str, int argc, char **argv) {
    int a;

    for (a = 1; a < argc; a++)
        if (!strcmp(str, argv[a]))
            return a;

    return -1;
}
void print_debug_info(const std::string &shard_path) {
    auto *store = singa::io::OpenStore("kvfile", shard_path, singa::io::kRead);
    std::cout << "xx" << std::endl;
    std::string key, value;

    std::ofstream out("debug_info.txt");

    DynamicRecord dynamicRecord;
    while (store->Read(&key, &value)) {
    dynamicRecord.ParseFromString(value);
    out << dynamicRecord.DebugString() << std::endl;
    }
    out.close();
}


int main(int argc, char **argv) {
    int i;
    FILE *f;

    // search for train file
    i = argPos(const_cast<char *>("-train"), argc, argv);
    if (i > 0) {
        if (i + 1 == argc) {
            printf("ERROR: training data file not specified!\n");
            return 0;
        }

        snprintf(train_file, strlen(argv[i + 1]) + 1, "%s", argv[i + 1]);

        f = fopen(train_file, "rb");
        if (f == NULL) {
            printf("ERROR: training data file not found!\n");
            return 0;
        }
        fclose(f);
    } else {
        printf("ERROR: training data must be set.\n");
    }

    // search for test file
    i = argPos(const_cast<char *>("-test"), argc, argv);
    if (i > 0) {
        if (i + 1 == argc) {
            printf("ERROR: testing data file not specified!\n");
            return 0;
        }

        snprintf(test_file, strlen(argv[i + 1]) + 1, "%s", argv[i + 1]);

        f = fopen(test_file, "rb");
        if (f == NULL) {
            printf("ERROR: testing data file not found!\n");
            return 0;
        }
        fclose(f);
        test_mode = 1;
    }

    create_data(train_file, "train_input_data.bin", "train_label_data.bin");
    if (test_mode) {
        create_data(test_file, "test_input_data.bin", "test_label_data.bin");
    }
    print_debug_info("test_input_data.bin");

    return 0;
}
