pgc++    -o result_nvpcomp29.out ../functionspace/antiInverseImage.cpp ../functionspace/distanceTransform.cpp ../functionspace/makeImageGrayScale.cpp ../functionspace/removeholesopt.cpp ../functionspace/thresholdMeasure.cpp ../functionspace/processImage.cpp ../functionspace/erosion.cpp ../functionspace/watershed.cpp  ../mainspace/WaterShedAlg.cpp `pkg-config opencv4 --cflags --libs`  -mp -Minfo=mp 
