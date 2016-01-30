/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/

#include <glog/logging.h>
#include <string>
#include "singa/singa.h"
#include "dpm.h"
#include "dpm.pb.h"

int main(int argc, char **argv) {
  // initialize glog before creating the driver
  google::InitGoogleLogging(argv[0]);
  
  singa::Driver driver;
  driver.Init(argc, argv);

  // if -resume in argument list, set resume to true; otherwise false
  int resume_pos = singa::ArgPos(argc, argv, "-resume");
  bool resume = (resume_pos != -1);

  // register all layers for dpm
  //driver.RegisterLayer<dpm::DynamicDataLayer, std::string>("kDynamicData");
  //driver.RegisterLayer<dpm::TimeSpanDataLayer, std::string>("kTimeSpanData");
  driver.RegisterLayer<dpm::DataLayer, std::string>("kData");
  driver.RegisterLayer<dpm::UnrollLayer, std::string>("kUnroll");
  driver.RegisterLayer<dpm::TimeUnrollLayer, std::string>("kTimeUnroll");
  driver.RegisterLayer<dpm::DPMGruLayer, std::string>("kDPMGru");
  driver.RegisterLayer<dpm::CombinationLayer, std::string>("kCombination");
  driver.RegisterLayer<dpm::DPMLabelLayer, std::string>("kDPMLabel"); // (batchsize, 1) for label info
  driver.RegisterLayer<dpm::DPMTimeLayer, std::string>("kDPMTime"); // (batchsize, 1) for delta_T info


  singa::JobProto jobConf = driver.job_conf();

  driver.Train(resume, jobConf);
  return 0;
}
