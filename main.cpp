//
//  main.cpp
//  Stitching_Detail
//
//  Created by KaiWu on May/13/15.
//  Copyright (c) 2015 KaiWu. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <ctime>
#include <random>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/util.hpp"

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/progress.hpp"


using namespace std;
using namespace cv;
using namespace cv::detail;
namespace fs = boost::filesystem;


int readImage(int, const char**);
void featureDetectionMatching(vector<Mat> &, int, vector< vector<KeyPoint> > &, vector<Mat> &, vector< vector<DMatch> > &);
void project2CylinderSphere(Mat &, float, float, int, Mat &);
void project2Cylinder(Mat &, float, float, Mat &);
void projectedImageSize(vector<Mat> &, vector<float> &, float, int &, int &);
void surface2plane(Point2f, float, float, int, int, int, int, Point2f &);
void imageRegistration(vector<Mat> &, vector<Mat> &, vector< vector<KeyPoint> > &, vector< vector<DMatch> > &, vector<float>, float, Mat &);
void getKeypoints(vector<KeyPoint>&, vector<KeyPoint> &, vector<DMatch> &, vector<Point2f> &, vector<Point2f> &);
void plane2surface(vector<Point2f> &, float, float, int, int, int, int);
void initParams(vector<Point2f>, vector<Point2f>, int, Mat &);
void initParams1(vector<Point2f>, vector<Point2f>, int, Mat &);
void initParamsRANSAC(vector<Point2f>, vector<Point2f>, int, Mat &);
void randomList(int, vector<int> &);
void lavmarBatch(vector<Point2f> &, vector<Point2f> &, Mat &);
void lavmarStochastic(vector<Point2f> &, vector<Point2f> &, Mat &);
void pointTransform(vector<Point2f> &, Mat &, vector<Point2f> &);
void imageTransform(Mat &, Mat &, Mat &, Point2i &, Mat &);
void findOverlappingArea(Mat &, Mat &, Mat &);
void findBoundingRect(Mat &, Point2i &, Point2i &);
void findSeam(Mat &, Mat &, Mat &, Mat &);
void blendImage(Mat &, Mat &, Mat &, Mat &);
void cameraCalibration(vector<Mat> &, vector<CameraParams> &);


const double PI = 3.1415926;
vector<string> imageNames;
vector<Mat> images;
bool try_gpu = false;
string features_type = "surf";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
float match_conf = 0.6f;
float conf_thresh = 1.f;


int main(int argc, const char * argv[]) {
    /*------Read in images------*/
    int retval = readImage(argc, argv);
    if(retval == -1)
    {
        return retval;
    }
    
    /*------Feature detection and matching------*/
    int imageNumber = static_cast<int>(images.size());
    
    vector< vector<KeyPoint> > keypoints;
    vector<Mat> descriptors(imageNumber);
    vector< vector<DMatch> > goodMatches(imageNumber - 1, vector<DMatch>(0));
    
    featureDetectionMatching(images, imageNumber, keypoints, descriptors, goodMatches);
    /*
    for(int i = 0; i < imageNumber; i++)
    {
        features[i].img_idx = i;
        features[i].img_size = Size(images[i].cols, images[i].rows);
        features[i].keypoints = keypoints[i];
        features[i].descriptors = descriptors[i];
    }
    */
    
    /*------Camera calibration (calculate focal length)------*/
    vector<CameraParams> cameras(imageNumber);
    cameraCalibration(images, cameras);
    
    /*------Set the median focal length as the scale of the warped image------*/
    vector<float> f(imageNumber);
    for(int i = 0; i < imageNumber; i++)
    {
        f[i] = cameras[i].focal;
    }
    sort(f.begin(), f.end());
    float warped_image_scale;
    if (f.size() % 2 == 1)
    {
        warped_image_scale = static_cast<float>(f[f.size() / 2]);
    }
    else
    {
        warped_image_scale = static_cast<float>(f[f.size() / 2 - 1] + f[f.size() / 2]) * 0.5;
    }
    
    /*------Project to cylindrical or spherical surface------*/
    int ind = 0; // 0 -> cylindrical; 1 -> spherical
    int projWidth = 0, projHeight = 0;
    vector<Mat> projImages(imageNumber);
    
    projectedImageSize(images, f, warped_image_scale, projWidth, projHeight); // all the images have the same sizes
    
    for(int i = 0; i < imageNumber; i++)
    {
        projImages[i] = Mat::zeros(projHeight, projWidth, CV_8U);
        project2CylinderSphere(images[i], f[i], warped_image_scale, ind, projImages[i]);
        //imshow(to_string(i), projImages[i]);
        //imwrite(to_string(i)+".jpg", projImages[i]);
    }

    /*------Image registration------*/
    Mat stitchedImage;
    imageRegistration(images, projImages, keypoints, goodMatches, f, warped_image_scale, stitchedImage);
    
    imwrite("stitched_image.jpg", stitchedImage);
    imshow("stitched image", stitchedImage);
    waitKey(0);
    
    return 0;
}

int readImage(int argc, const char** argv) {
    string imageName;
    
    boost::progress_timer t(std::clog);
    
    fs::path full_path(fs::initial_path<fs::path>());
    
    if (argc > 1)
        full_path = fs::system_complete(fs::path(argv[1]));
    else
        std::cout << "\nusage:   simple_ls [path]" << std::endl;
    
    unsigned long file_count = 0;
    unsigned long dir_count = 0;
    unsigned long other_count = 0;
    unsigned long err_count = 0;
    string fileName;
    
    if (!fs::exists(full_path))
    {
        std::cout << "\nNot found: " << full_path.string() << std::endl;
        return -1;
    }
    
    if (fs::is_directory(full_path))
    {
        std::cout << "\nIn directory: " << full_path.string() << "\n\n";
        fs::directory_iterator end_iter;
        for (fs::directory_iterator dir_itr(full_path); dir_itr != end_iter; ++dir_itr)
        {
            try
            {
                if (fs::is_directory(dir_itr->status()))
                {
                    ++dir_count;
                    std::cout << dir_itr->path().filename() << " [directory]\n";
                }
                else if (fs::is_regular_file(dir_itr->status()))
                {
                    fileName = dir_itr->path().filename().string();
                    if(fileName.find(".png") != string::npos || fileName.find(".jpg") != string::npos)
                    {
                        ++file_count;
                        imageName = string(argv[1]).append("/").append(fileName);
                        if(!imageName.empty()) {
                            imageNames.push_back(imageName);
                        }
                        images.push_back(imread(imageName, CV_LOAD_IMAGE_GRAYSCALE));
                    }
                }
                else
                {
                    ++other_count;
                    std::cout << dir_itr->path().filename() << " [other]\n";
                }
            }
            catch (const std::exception & ex)
            {
                ++err_count;
                std::cout << dir_itr->path().filename() << " " << ex.what() << std::endl;
            }
        }
        std::cout << "\n" << file_count << " files\n";
        //<< dir_count << " directories\n"
        //<< other_count << " others\n"
        //<< err_count << " errors\n";
    }
    else // must be a file
    {
        std::cout << "\nFound: " << full_path.string() << "\n";
    }
    
    return 0;
}

void featureDetectionMatching(vector<Mat> & images, int imageNumber, vector< vector<KeyPoint> > & keypoints, vector<Mat> & descriptors, vector< vector<DMatch> > & goodMatches)
{
    //-- Detect the keypoints using SURF Detector
    int minHessian = 400;
    
    SurfFeatureDetector detector( minHessian );
    detector.detect(images, keypoints);
    
    //-- Calculate descriptors (feature vectors)
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SURF");
    extractor->compute(images, keypoints, descriptors);
    
    //-- Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    //vector< vector<DMatch> > matches(imageNumber - 1, vector<DMatch>(0));
    vector< vector< vector<DMatch> > > matches(imageNumber - 1, vector< vector<DMatch> >(0));
    for(int i = 0; i < imageNumber - 1; i++)
    {
        //matcher.match(descriptors[i], descriptors[i + 1], matches[i]);
        matcher.knnMatch(descriptors[i], descriptors[i + 1], matches[i], 2);
    }
    
    const float ratio = 0.8; // 0.8 in Lowe's paper; can be tuned
    for(int i = 0; i < imageNumber - 1; i++)
    {
        for (int j = 0; j < static_cast<int>(matches[i].size()); j++)
        {
            if (matches[i][j][0].distance < ratio * matches[i][j][1].distance)
            {
                goodMatches[i].push_back(matches[i][j][0]);
            }
        }
    }
    /*
    //-- Quick calculation of max and min distances between keypoints
    vector<double> max_dist(imageNumber - 1, double(0));
    vector<double> min_dist(imageNumber - 1, double(100));
    double dist;
    
    for(int i = 0; i < imageNumber - 1; i++)
    {
        for(int j = 0; j < static_cast<int>(matches[i].size()); j++)
        {
            dist = matches[i][j].distance;
            if(dist != 0 && dist < min_dist[i]) min_dist[i] = dist;
            if(dist != 0 && dist > max_dist[i]) max_dist[i] = dist;
        }
    }*/
    /*
    for(int i = 0; i < imageNumber - 1; i++)
    {
        printf("-- Max dist for the %d and %d image: %f \n", i + 1, i + 2, max_dist[i]);
        printf("-- Min dist for the %d and %d image: %f \n", i + 1, i + 2, min_dist[i]);
    }*/
    /*
    //-- Use only "good" matches (i.e. whose distance is less than 3*min_dist )
    for(int i = 0; i < imageNumber - 1; i++)
    {
        for(int j = 0; j < static_cast<int>(matches[i].size()); j++)
        {
            if(matches[i][j].distance < 3 * min_dist[i])
            {
                goodMatches[i].push_back(matches[i][j]);
            }
        }
    }*/
    /*
    //-- Draw only "good" matches
    Mat img_matches;
    for(int i = 0; i < imageNumber - 1; i++)
    {
        drawMatches(images[i], keypoints[i], images[i + 1], keypoints[i + 1], goodMatches[i], img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        //-- Show detected matches
        imshow(to_string(i), img_matches);
    }*/
}

void project2CylinderSphere(Mat & image, float f, float s, int ind, Mat & projImage)
{
    switch (ind) {
        case 0:
            project2Cylinder(image, f, s, projImage);
            break;
        case 1:
        default:
            cout << "0 -> cylindrical, 1 -> spherical" << endl;
    }
}

void project2Cylinder(Mat & image, float focal, float s, Mat & projImage)
{
    int width = 0, height = 0, projWidth = projImage.cols, projHeight = projImage.rows;
    
    width = image.cols;
    height = image.rows;
    
    for(int r = 0; r < projHeight; r++)
    {
        for(int c = 0; c < projWidth; c++)
        {
            Point2f current_pos(c, r);
            surface2plane(current_pos, focal, s, width, height, projWidth, projHeight, current_pos);
            
            Point2i top_left((int)current_pos.x, (int)current_pos.y); //top left because of integer rounding
            
            //make sure the point is actually inside the original image
            if(top_left.x < 0 ||
               top_left.x > width-2 ||
               top_left.y < 0 ||
               top_left.y > height-2)
            {
                projImage.at<uchar>(r, c) = 0;
                continue;
            }
            
            //bilinear interpolation
            float dx = current_pos.x - top_left.x;
            float dy = current_pos.y - top_left.y;
            
            float weight_tl = (1.0 - dx) * (1.0 - dy);
            float weight_tr = (dx)       * (1.0 - dy);
            float weight_bl = (1.0 - dx) * (dy);
            float weight_br = (dx)       * (dy);
            
            uchar value = weight_tl * image.at<uchar>(top_left) +
            weight_tr * image.at<uchar>(top_left.y,top_left.x+1) +
            weight_bl * image.at<uchar>(top_left.y+1,top_left.x) +
            weight_br * image.at<uchar>(top_left.y+1,top_left.x+1);
            
            projImage.at<uchar>(r, c) = value;
        }
    }
}

void projectedImageSize(vector<Mat> & images, vector<float> & f, float s, int & width, int & height)
{
    int imageNumber = static_cast<int>(images.size());
    
    for(int i = 0; i < imageNumber; i++)
    {
        if(width < ceil(images[i].cols / f[i] * s))
        {
            width = ceil(images[i].cols / f[i] * s);
        }
        if(height < ceil(images[i].rows / f[i] * s))
        {
            height = ceil(images[i].rows / f[i] * s);
        }
    }
}

void surface2plane(Point2f point, float f, float s, int width, int height, int projWidth, int projHeight, Point2f & finalPoint)
{
    float halfWidth = float(width) / 2;
    float halfHeight = float(height) / 2;
    float halfProjWidth = float(projWidth) / 2;
    float halfProjHeight = float(projHeight) / 2;
    
    float theta = (point.x - halfProjWidth) / s;
    float h = (point.y - halfProjHeight) / s;
    
    finalPoint.x = halfWidth + f * tan(theta);
    finalPoint.y = halfHeight + h * sqrt((point.x - halfWidth) * (point.x - halfWidth) + f * f);
}

void imageRegistration(vector<Mat> & images, vector<Mat> & projImages, vector< vector<KeyPoint> > & keypoints, vector< vector<DMatch> > & pairwise_matches, vector<float> f, float s, Mat & stitchedImage)
{
    vector<KeyPoint> keypointsSource, keypointsTarget;
    vector<DMatch> matches;
    int width, height, projWidth = 0, projHeight = 0;
    int imageNumber = static_cast<int>(images.size());
    int matchesNumber = 0;
    Mat motionParams(3, 1, CV_32F);
    Mat compositeImage;
    vector<Point2f> pointSource, pointTarget;
    Point2i margin(0, 0);
    Mat maskCompositeImage, maskTransformedImage;
    
    projectedImageSize(images, f, s, projWidth, projHeight); // all the images have the same sizes
    
    compositeImage = projImages[0].clone();
    
    for(int i = 0; i < imageNumber - 1; i++)
    {
        keypointsSource = keypoints[i + 1];
        keypointsTarget = keypoints[i];
        matches = pairwise_matches[i];
        matchesNumber = static_cast<int>(matches.size());
        
        getKeypoints(keypointsSource, keypointsTarget, matches, pointSource, pointTarget);
        
        width = images[i].cols; height = images[i].rows;
        plane2surface(pointTarget, f[i], s, width, height, projWidth, projHeight);
        width = images[i + 1].cols; height = images[i + 1].rows;
        plane2surface(pointSource, f[i + 1], s, width, height, projWidth, projHeight);
        /*
         int num = static_cast<int>(pointTarget.size());
         Mat imageSource = projImages[i + 1].clone();
         Mat imageTarget = projImages[i].clone();
         
         Mat img_matches;
         drawMatches(imageSource, keypointsSource, imageTarget, keypointsTarget, matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
         imshow(to_string(i), img_matches);
         
         for(int j = 0; j < num; j++)
         {
         cv::circle(imageSource, pointSource[j], 1, Scalar(255, 0, 0));
         cv::circle(imageTarget, pointTarget[j], 1, Scalar(255, 0, 0));
         }
         imshow("source image: " + to_string(i + 1), imageSource);
         imshow("target image: " + to_string(i), imageTarget);
         */
        if(i != 0)
        {
            pointTransform(pointTarget, motionParams, pointTarget);
            /*
             Mat tmpImage = compositeImage.clone();
             for(int j = 0; j < num; j++)
             {
             cv::circle(tmpImage, pointTarget[j], 1, Scalar(255, 0, 0));
             }
             imshow("composite image 0 - " + to_string(i), tmpImage);*/
        }
        
        initParamsRANSAC(pointSource, pointTarget, matchesNumber, motionParams);
        
        //lavmarBatch(pointSource, pointTarget, motionParams);
        //lavmarStochastic(pointSource, pointTarget, motionParams);
        
        imageTransform(compositeImage, projImages[i + 1], motionParams, margin, stitchedImage);
        compositeImage = stitchedImage.clone();
        
        pointSource.clear();
        pointTarget.clear();
    }
}

void getKeypoints(vector<KeyPoint>& keypointsSource, vector<KeyPoint> & keypointsTarget, vector<DMatch> & matches, vector<Point2f> & pointSource, vector<Point2f> & pointTarget)
{
    Point2f ptSource, ptTarget;
    int matchesNumber = static_cast<int>(matches.size());
    for(int j = 0; j < matchesNumber; j++)
    {
        ptSource = keypointsSource[matches[j].trainIdx].pt;
        ptTarget = keypointsTarget[matches[j].queryIdx].pt;
        pointSource.push_back(ptSource);
        pointTarget.push_back(ptTarget);
    }
}

void plane2surface(vector<Point2f> & point, float f, float s, int width, int height, int projWidth, int projHeight)
{
    float halfWidth = float(width) / 2;
    float halfHeight = float(height) / 2;
    float halfProjWidth = float(projWidth) / 2;
    float halfProjHeight = float(projHeight) / 2;
    float theta, h;
    int pointNumber = static_cast<int>(point.size());
    
    for(int i = 0; i < pointNumber; i++)
    {
        theta = atan((point[i].x - halfWidth) / f);
        h = (point[i].y - halfHeight) / sqrt((point[i].x - halfWidth) * (point[i].x - halfWidth) + f * f);
        point[i].x = s * theta + halfProjWidth;
        point[i].y = s * h + halfProjHeight;
    }
}

/* Estimate the Motion Parameters of Similarity Transformation */
void initParams(vector<Point2f> ptSource, vector<Point2f> ptTarget, int matchesNumber, Mat & motionParams)
{
    Mat p = Mat::zeros(4, 1, CV_32F),
    A = Mat::zeros(4, 4, CV_32F),
    b = Mat::zeros(4, 1, CV_32F),
    J = Mat::zeros(2, 4, CV_32F),
    r = Mat::zeros(2, 1, CV_32F);
    
    for(int i = 0; i < matchesNumber; i++)
    {
        J = (Mat_<float>(2, 4) << 1, 0, ptSource[i].x, -ptSource[i].y, 0, 1, ptSource[i].y, ptSource[i].x);
        r = (Mat_<float>(2, 1) << ptTarget[i].x - ptSource[i].x, ptTarget[i].y - ptSource[i].y);
        
        A = A + J.t() * J;
        b = b + J.t() * r;
    }
    p = A.inv(DECOMP_SVD) * b;
    //cout << "p: " << p << endl;
    motionParams.at<float>(0, 0) = p.at<float>(0, 0);
    motionParams.at<float>(1, 0) = p.at<float>(1, 0);
    motionParams.at<float>(2, 0) = atan(p.at<float>(3, 0) / (1 + p.at<float>(2, 0)));
}

/* Least-Squares Rigid Motion Using SVD */
void initParams1(vector<Point2f> ptSource, vector<Point2f> ptTarget, int matchesNumber, Mat & motionParams)
{
    Point2f ptCenterSource(0, 0), ptCenterTarget(0, 0);
    Mat X(2, matchesNumber, CV_32F), Y(matchesNumber, 2, CV_32F);
    
    for(int i = 0; i < matchesNumber; i++)
    {
        ptCenterSource.x = ptCenterSource.x + ptSource[i].x;
        ptCenterSource.y = ptCenterSource.y + ptSource[i].y;
        ptCenterTarget.x = ptCenterTarget.x + ptTarget[i].x;
        ptCenterTarget.y = ptCenterTarget.y + ptTarget[i].y;
    }
    ptCenterSource.x = ptCenterSource.x / matchesNumber;
    ptCenterSource.y = ptCenterSource.y / matchesNumber;
    ptCenterTarget.x = ptCenterTarget.x / matchesNumber;
    ptCenterTarget.y = ptCenterTarget.y / matchesNumber;
    
    for(int i = 0; i < matchesNumber; i++)
    {
        X.at<float>(0, i) = ptSource[i].x - ptCenterSource.x;
        X.at<float>(1, i) = ptSource[i].y - ptCenterSource.y;
        Y.at<float>(i, 0) = ptTarget[i].x - ptCenterTarget.x;
        Y.at<float>(i, 1) = ptTarget[i].y - ptCenterTarget.y;
    }
    
    Mat S, w, u, vt;
    S = X * Y;
    SVD::compute(S, w, u, vt);
    
    Mat Sigma = Mat::eye(2, 2, CV_32F);
    Sigma.at<float>(1, 1) = determinant((u * vt).t());
    //cout << "Sigma: " << Sigma << endl;
    
    Mat R = vt.t() * Sigma * u.t();
    //cout << "R: " << R << endl;
    
    motionParams.at<float>(0, 0) = ptCenterTarget.x - (R.at<float>(0, 0) * ptCenterSource.x + R.at<float>(0, 1) * ptCenterSource.y);
    motionParams.at<float>(1, 0) = ptCenterTarget.y - (R.at<float>(1, 0) * ptCenterSource.x + R.at<float>(1, 1) * ptCenterSource.y);
    motionParams.at<float>(2, 0) = atan((-R.at<float>(0, 1) + R.at<float>(1, 0)) / (R.at<float>(0, 0) + R.at<float>(1, 1)));
}

void initParamsRANSAC(vector<Point2f> ptSource, vector<Point2f> ptTarget, int matchesNumber, Mat & motionParams)
{
    float theta;
    float dist, minDist = INFINITY, sumDist = 0, meanDist = 0, minMeanDist = INFINITY;
    int k = 100, inlierNumber = 0, trainNumber = 2; // minimum number of training examples?
    vector<int> randList;
    vector<Point2f> trainPtSource, trainPtTarget;
    Mat J(2, 3, CV_32F), T(2, 3, CV_32F),
    r(2, 1, CV_32F), g(3, 1, CV_32F),
    ptStart(3, 1, CV_32F), ptEnd(2, 1, CV_32F),
    motionParamsOpt(motionParams);
    
    while(k--)
    {
        inlierNumber = 0;
        minDist = INFINITY;
        sumDist = 0;
        
        randomList(matchesNumber, randList);
        
        for(int i = 0; i < trainNumber; i++)
        {
            trainPtSource.push_back(ptSource[randList[i]]);
            trainPtTarget.push_back(ptTarget[randList[i]]);
        }
        
        initParams1(trainPtSource, trainPtTarget, trainNumber, motionParams);
        theta = motionParams.at<float>(2, 0);
        
        for(int i = 0; i < matchesNumber; i++)
        {
            ptStart = (Mat_<float>(3, 1) << ptSource[randList[i]].x, ptSource[randList[i]].y, 1);
            ptEnd = (Mat_<float>(2, 1) << ptTarget[randList[i]].x, ptTarget[randList[i]].y);
            J = (Mat_<float>(2, 3) << 1, 0, -sin(theta) * ptSource[randList[i]].x - cos(theta) * ptSource[randList[i]].y, 0, 1, cos(theta) * ptSource[randList[i]].x - sin(theta) * ptSource[randList[i]].y);
            T = (Mat_<float>(2, 3) << cos(theta), -sin(theta), motionParams.at<float>(0, 0), sin(theta), cos(theta), motionParams.at<float>(1, 0));
            r = ptEnd - T * ptStart;
            g = J.t() * r;
            dist = norm(g);
            
            if(i < trainNumber)
            {
                if(dist < minDist)
                {
                    minDist = dist;
                }
            }
            else
            {
                if(dist <= 3 * minDist)
                {
                    sumDist = sumDist + dist;
                    inlierNumber++;
                }
            }
        }
        
        meanDist = sumDist / inlierNumber;
        if(meanDist < minMeanDist)
        {
            motionParamsOpt = motionParams.clone(); //motionParams.copyTo(motionParamsOpt);
            minMeanDist = meanDist;
        }
        
        trainPtSource.erase(trainPtSource.begin(), trainPtSource.end());
        trainPtTarget.erase(trainPtTarget.begin(), trainPtTarget.end());
        randList.erase(randList.begin(), randList.end());
    }
    motionParams = motionParamsOpt.clone(); //motionParamsOpt.copyTo(motionParams);
    cout << "initial motionParams: " << motionParamsOpt << endl << "min mean dist: " << minMeanDist << endl;
}

void randomList(int size, vector<int> & elements)
{
    for(int i = 0; i < size; i++)
    {
        elements.push_back(i);
    }
    
    //cout << "Before: ";
    //copy(elements.cbegin(), elements.cend(), std::ostream_iterator<int>(std::cout, " "));
    
    int currentIndexCounter = static_cast<int>(elements.size()) - 1;
    
    for (int i = currentIndexCounter; i > 0; i--)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, i);
        int randomIndex = dis(gen);
        
        if (randomIndex != i)
        {
            swap(elements.at(randomIndex), elements.at(i));
        }
    }
    
    //cout << "\nAfter: ";
    //copy(elements.cbegin(), elements.cend(), std::ostream_iterator<int>(std::cout, " "));
    //cout << endl;
}

void lavmarBatch(vector<Point2f> & pointSource, vector<Point2f> & pointTarget, Mat & motionParams)
{
    Mat A = Mat::zeros(3, 3, CV_32F), g = Mat::zeros(3, 1, CV_32F);
    Mat J(2, 3, CV_32F), r(2, 1, CV_32F), sumR(2, 1, CV_32F), rNew(2, 1, CV_32F), sumRNew(2, 1, CV_32F);
    Mat motionParamsNew(3, 1, CV_32F), dp(3, 1, CV_32F);
    Mat T(2, 3, CV_32F);
    Mat ptStart(3, 1, CV_32F), ptEnd(2, 1, CV_32F);
    
    bool stop = false;
    int v = 2, beta = 2, k = 0, maxIter = 100, matchesNumber = static_cast<int>(pointSource.size());
    double delta = 3, p = 3;
    float mu, tau = 1e-3, thre = 1e-15, rho = 0, theta = motionParams.at<float>(2, 0); // mu: damping parameter
    
    for(int j = 0; j < matchesNumber; j++)
    {
        J = (Mat_<float>(2, 3) << 1, 0, -sin(theta) * pointSource[j].x - cos(theta) * pointSource[j].y, 0, 1, cos(theta) * pointSource[j].x - sin(theta) * pointSource[j].y);
        T = (Mat_<float>(2, 3) << cos(theta), -sin(theta), motionParams.at<float>(0, 0), sin(theta), cos(theta), motionParams.at<float>(1, 0));
        ptStart = (Mat_<float>(3, 1) << pointSource[j].x, pointSource[j].y, 1);
        ptEnd = (Mat_<float>(2, 1) << pointTarget[j].x, pointTarget[j].y);
        
        r = ptEnd - T * ptStart;
        //cout << "J: " << J << endl << "r: " << r << endl;
        
        A = A + J.t() * J;
        g = g + J.t() * r;
        sumR = sumR + r;
    }
    
    stop = (norm(g, NORM_INF) < thre);
    double minDiag, maxDiag;
    minMaxLoc(A.diag(), &minDiag, &maxDiag);
    mu = tau * (float)maxDiag;
    
    while(!stop && k < maxIter)
    {
        k++;
        rho = 0;
        while(!(rho > 0 || stop)) // rho <= 0 && not stop, execute mu *= v, v *= 2;
        {
            dp = (A + mu * Mat::eye(Size(A.cols, A.rows), CV_32F)).inv(cv::DECOMP_SVD) * g;
            if(norm(dp) <= thre * norm(motionParams))
            {
                stop = true;
            }
            else
            {
                motionParamsNew = motionParams + dp;
                theta = motionParamsNew.at<float>(2, 0);
                
                for(int j = 0; j < matchesNumber; j++)
                {
                    J = (Mat_<float>(2, 3) << 1, 0, -sin(theta) * pointSource[j].x - cos(theta) * pointSource[j].y, 0, 1, cos(theta) * pointSource[j].x - sin(theta) * pointSource[j].y);
                    T = (Mat_<float>(2, 3) << cos(theta), -sin(theta), motionParamsNew.at<float>(0, 0), sin(theta), cos(theta), motionParamsNew.at<float>(1, 0));
                    ptStart = (Mat_<float>(3, 1) << pointSource[j].x, pointSource[j].y, 1);
                    ptEnd = (Mat_<float>(2, 1) << pointTarget[j].x, pointTarget[j].y);
                    
                    rNew = ptEnd - T * ptStart;
                    //cout << "J: " << J << endl << "r: " << rNew << endl;
                    
                    A = A + J.t() * J;
                    g = g + J.t() * rNew;
                    sumRNew = sumRNew + rNew;
                }
                rho = (pow(norm(sumR), 2.0) - pow(norm(sumRNew), 2.0)) / norm(dp.t() * (mu * dp + g));
                if(rho > 0)
                {
                    motionParams = motionParamsNew.clone();
                    r = rNew.clone();
                    stop = norm(g, NORM_INF) < thre | norm(r) < thre;
                    mu = mu * max(1.0 / delta, 1 - (beta - 1) * pow((2 * rho - 1), p));
                    v = (int)beta;
                }
                else
                {
                    mu = mu * v;
                    v = 2 * v;
                }
            }
        }
    }
    cout << "final motionParams: " << motionParams << endl;
}

void lavmarStochastic(vector<Point2f> & pointSource, vector<Point2f> & pointTarget, Mat & motionParams)
{
    Mat A = Mat::zeros(3, 3, CV_32F), g = Mat::zeros(3, 1, CV_32F);
    Mat J(2, 3, CV_32F), r(2, 1, CV_32F), rNew(2, 1, CV_32F);
    Mat motionParamsNew(3, 1, CV_32F), dp(3, 1, CV_32F);
    Mat T(2, 3, CV_32F);
    Mat ptStart(3, 1, CV_32F), ptEnd(2, 1, CV_32F);
    
    bool stop = false;
    int v = 2, beta = 2, k = 0, maxIter = 100, matchesNumber = static_cast<int>(pointSource.size());
    double delta = 3, p = 3;
    float theta, mu, tau = 1e-3, thre = 1e-15, rho = 0; // mu: damping parameter
    
    theta = motionParams.at<float>(2, 0);
    srand(static_cast<unsigned int>(time(nullptr)));
    int ind = rand() % matchesNumber;
    J = (Mat_<float>(2, 3) << 1, 0, -sin(theta) * pointSource[ind].x - cos(theta) * pointSource[ind].y, 0, 1, cos(theta) * pointSource[ind].x - sin(theta) * pointSource[ind].y);
    T = (Mat_<float>(2, 3) << cos(theta), -sin(theta), motionParams.at<float>(0, 0), sin(theta), cos(theta), motionParams.at<float>(1, 0));
    ptStart = (Mat_<float>(3, 1) << pointSource[ind].x, pointSource[ind].y, 1);
    ptEnd = (Mat_<float>(2, 1) << pointTarget[ind].x, pointTarget[ind].y);
    r = ptEnd - T * ptStart;
    A = J.t() * J;
    g = J.t() * r;
    
    stop = (norm(g, NORM_INF) < thre);
    double minDiag, maxDiag;
    minMaxLoc(A.diag(), &minDiag, &maxDiag);
    mu = tau * (float)maxDiag;
    
    while(!stop && k < maxIter)
    {
        k++;
        rho = 0;
        while(!(rho > 0 || stop)) // rho <= 0 && not stop, execute mu *= v, v *= 2;
        {
            dp = (A + mu * Mat::eye(Size(A.cols, A.rows), CV_32F)).inv(cv::DECOMP_SVD) * g;
            if(norm(dp) <= thre * norm(motionParams))
            {
                stop = true;
            }
            else
            {
                motionParamsNew = motionParams + dp;
                
                theta = motionParamsNew.at<float>(2, 0);
                ind = rand() % matchesNumber;
                J = (Mat_<float>(2, 3) << 1, 0, -sin(theta) * pointSource[ind].x - cos(theta) * pointSource[ind].y, 0, 1, cos(theta) * pointSource[ind].x - sin(theta) * pointSource[ind].y);
                T = (Mat_<float>(2, 3) << cos(theta), -sin(theta), motionParamsNew.at<float>(0, 0), sin(theta), cos(theta), motionParamsNew.at<float>(1, 0));
                ptStart = (Mat_<float>(3, 1) << pointSource[ind].x, pointSource[ind].y, 1);
                ptEnd = (Mat_<float>(2, 1) << pointTarget[ind].x, pointTarget[ind].y);
                rNew = ptEnd - T * ptStart;
                A = J.t() * J;
                g = J.t() * rNew;
                
                rho = (pow(norm(r), 2.0) - pow(norm(rNew), 2.0)) / norm(dp.t() * (mu * dp + g));
                //cout << "check out: " << dp.t() * (mu * dp + g) << endl;
                if(rho > 0)
                {
                    motionParamsNew.copyTo(motionParams);
                    rNew.copyTo(r);
                    stop = norm(g, NORM_INF) < thre | norm(r) < thre;
                    mu = mu * max(1 / delta, 1 - (beta - 1) * pow((2 * rho - 1), p));
                    v = beta;
                }
                else
                {
                    mu = mu * v;
                    v = 2 * v;
                }
            }
        }
    }
    cout << "final motionParams: " << motionParams << endl;
}

void pointTransform(vector<Point2f> & inputPoint, Mat & motionParams, vector<Point2f> & outputPoint)
{
    Mat T(2, 3, CV_32F), inputPt(3, 1, CV_32F), outputPt(2, 1, CV_32F);
    float theta = motionParams.at<float>(2, 0);
    T = (Mat_<float>(2, 3) << cos(theta), -sin(theta), motionParams.at<float>(0, 0), sin(theta), cos(theta), motionParams.at<float>(1, 0));
    
    for(int i = 0; i < static_cast<int>(inputPoint.size()); i++)
    {
        inputPt = (Mat_<float>(3, 1) << inputPoint[i].x, inputPoint[i].y, 1);
        outputPt = T * inputPt;
        outputPoint[i].x = outputPt.at<float>(0, 0);
        outputPoint[i].y = outputPt.at<float>(1, 0);
    }
}

void imageTransform(Mat & compositeImage, Mat & image, Mat & motionParams, Point2i & marginTB, Mat & stitchedImage)
{
    Mat transformedImage, roi;
    int width = image.cols, height = image.rows;
    int compositeWidth, compositeHeight;
    int marginTop = marginTB.x, marginBottom = marginTB.y, marginDiff = 0;
    int t_x = floor(motionParams.at<float>(0, 0));
    float margin = 0, theta = motionParams.at<float>(2, 0);
    Point2f origin(0, 0);
    
    // extend along the x-axis
    margin = fabs(height * sin(theta));
    if(theta > 0)
    {
        origin.x = ceil(margin);
    }
    else
    {
        origin.x = 0;
    }
    compositeWidth = origin.x + width + t_x + 1;
    
    // extend along the y-axis
    margin = width * sin(theta) + motionParams.at<float>(1, 0);
    if(margin < 0)
    {
        int marginTopCurr = ceil(fabs(margin));
        if(marginTop < marginTopCurr)
        {
            marginDiff = marginTopCurr - marginTop;
            marginTop = marginTopCurr;
            marginTB.x = marginTop;
        }
        origin.y = marginTop;
    }
    else
    {
        if(marginBottom < ceil(margin))
        {
            marginBottom = ceil(margin);
            marginTB.y = marginBottom;
        }
    }
    compositeHeight = height + marginTop + marginBottom;
    stitchedImage = Mat::zeros(compositeHeight, compositeWidth, CV_8U);
    roi = stitchedImage(Rect(0, marginDiff, compositeImage.cols, compositeImage.rows));
    compositeImage.copyTo(roi);
    
    transformedImage = Mat::zeros(compositeHeight, compositeWidth, CV_8U);
    roi = transformedImage(Rect(origin.x, origin.y, width, height));
    image.copyTo(roi);
    
    Mat rotation(2, 3, CV_32F), translation(2, 3, CV_32F);
    rotation = getRotationMatrix2D(origin, -theta / PI * 180, 1);
    warpAffine(transformedImage, transformedImage, rotation, transformedImage.size());
    
    translation = (Mat_<float>(2,3) << 1, 0, motionParams.at<float>(0, 0) - origin.x, 0, 1, motionParams.at<float>(1, 0));
    warpAffine(transformedImage, transformedImage, translation, transformedImage.size());
    
    Mat overlapReg1 = stitchedImage(Rect(t_x - origin.x, 0, compositeWidth - 1 - (t_x - origin.x), compositeHeight));
    Mat overlapReg2 = transformedImage(Rect(t_x - origin.x, 0, compositeWidth - 1 - (t_x - origin.x), compositeHeight));
    Mat overlappingArea = Mat::zeros(compositeHeight, compositeWidth - 1 - (t_x - origin.x), CV_8U);
    findOverlappingArea(overlapReg1, overlapReg2, overlappingArea);
    
    Mat seam; // find the coordinates of the seam points
    findSeam(overlapReg1, overlapReg2, overlappingArea, seam);
    
    blendImage(overlapReg1, overlapReg2, overlappingArea, seam);
    //add(overlapReg1, overlapReg2, overlapReg1, ~overlappingArea);
}

void findOverlappingArea(Mat & compositeImage, Mat & stitchedImage, Mat & overlappingArea)
{
    int width = static_cast<int>(compositeImage.cols), height = static_cast<int>(compositeImage.rows);
    
    for(int r = 0; r < height; r++)
    {
        for(int c = 0; c < width; c++)
        {
            if(int(compositeImage.at<uchar>(r, c)) > 0 && int(stitchedImage.at<uchar>(r, c)) > 0)
            {
                overlappingArea.at<uchar>(r, c) = 255;
            }
        }
    }
}

void findSeam(Mat & image1, Mat & image2, Mat & overlappingArea, Mat & seam)
{
    Point2i tr, br;
    findBoundingRect(overlappingArea, tr, br); // find the bounding rectangle of the overlapping area, top right and top left points
    
    Mat S(image1.rows, br.x + 1, CV_16U);
    int maxValue = 65535;
    S.setTo(maxValue);
    
    seam = Mat::zeros(image1.rows, 1, CV_32S);
    
    for(int r = tr.y; r <= br.y; r++)
    {
        for(int c = 0; c <= tr.x; c++)
        {
            if((int)overlappingArea.at<uchar>(r, c) == 255)
            {
                if(r == tr.y || (S.at<ushort>(r - 1, c - 1) == S.at<ushort>(r - 1, c) == S.at<ushort>(r - 1, c + 1) == maxValue))
                {
                    S.at<ushort>(r, c) = abs(image1.at<uchar>(r, c) - image2.at<uchar>(r, c));
                }
                else
                {
                    if(c == 0)
                    {
                        S.at<ushort>(r, c) = abs(image1.at<uchar>(r, c) - image2.at<uchar>(r, c)) + min(S.at<ushort>(r - 1, c), S.at<ushort>(r - 1, c + 1));
                    }
                    else if(c == tr.x)
                    {
                        S.at<ushort>(r, c) = abs(image1.at<uchar>(r, c) - image2.at<uchar>(r, c)) + min(S.at<ushort>(r - 1, c - 1), S.at<ushort>(r - 1, c));
                    }
                    else
                    {
                        S.at<ushort>(r, c) = abs(image1.at<uchar>(r, c) - image2.at<uchar>(r, c)) + min(S.at<ushort>(r - 1, c - 1), min(S.at<ushort>(r - 1, c), S.at<ushort>(r - 1, c + 1)));
                    }
                }
            }
        }
    }
    
    int minIdx[2];
    minMaxIdx(S.row(br.y), NULL, NULL, minIdx);
    seam.at<int>(br.y, 0) = minIdx[1];
    
    int col = minIdx[1];
    for(int r = br.y - 1; r >= tr.y; r--)
    {
        int c = col;
        if(S.at<ushort>(r, c - 1) < S.at<ushort>(r, c) && S.at<ushort>(r, c - 1) < S.at<ushort>(r, c + 1))
        {
            col = c - 1;
        }
        else if(S.at<ushort>(r, c) < S.at<ushort>(r, c - 1) && S.at<ushort>(r, c) < S.at<ushort>(r, c + 1))
        {
            col = c;
        }
        else if(S.at<ushort>(r, c + 1) < S.at<ushort>(r, c - 1) && S.at<ushort>(r, c + 1) < S.at<ushort>(r, c))
        {
            col = c + 1;
        }
        seam.at<int>(r, 0) = col;
    }
}

void findBoundingRect(Mat & overlappingArea, Point2i & tr, Point2i & br)
{
    int w = overlappingArea.cols;
    int h = overlappingArea.rows;
    int minRow = h, maxRow = 0, minCol = w, maxCol = 0;
    
    for(int r = 0; r < h; r++)
    {
        for(int c = 0; c < w; c++)
        {
            if((int)overlappingArea.at<uchar>(r, c) == 255)
            {
                if(r < minRow)
                    minRow = r;
                if(r > maxRow)
                    maxRow = r;
                if(c < minCol)
                    minCol = c;
                if(c > maxCol)
                    maxCol = c;
            }
        }
    }
    tr.x = maxCol;
    tr.y = minRow;
    br.x = maxCol;
    br.y = maxRow;
}

void blendImage(Mat & stitchedImage, Mat & transformedImage, Mat & overlappingArea, Mat & seam)
{
    int height = static_cast<int>(stitchedImage.rows);
    int width = static_cast<int>(stitchedImage.cols);
    
    for(int r = 0; r < height; r++)
    {
        for(int c = 0; c < width; c++)
        {
            if(((int)overlappingArea.at<uchar>(r, c) == 255 && c >= seam.at<int>(r, 1)))
            {
                stitchedImage.at<uchar>(r, c) = transformedImage.at<uchar>(r, c);
            }
            else if((int)overlappingArea.at<uchar>(r, c) != 255)
            {
                stitchedImage.at<uchar>(r, c) = max(stitchedImage.at<uchar>(r, c), transformedImage.at<uchar>(r, c));
            }
        }
    }
}

void cameraCalibration(vector<Mat> & images, vector<CameraParams> & cameras)
{
    int num_images = static_cast<int>(images.size());
    
    Ptr<FeaturesFinder> finder;
    if (features_type == "surf")
    {
#if defined(HAVE_OPENCV_NONFREE) && defined(HAVE_OPENCV_GPU)
        if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
            finder = new SurfFeaturesFinderGpu();
        else
#endif
            finder = new SurfFeaturesFinder();
    }
    else if (features_type == "orb")
    {
        finder = new OrbFeaturesFinder();
    }
    else
    {
        cout << "Unknown 2D features type: '" << features_type << "'.\n";
    }
    
    vector<ImageFeatures> features(num_images);
    
    for (int i = 0; i < num_images; ++i)
    {
        (*finder)(images[i], features[i]);
        features[i].img_idx = i;
    }
    finder->collectGarbage();
    
    vector<MatchesInfo> pairwise_matches;
    BestOf2NearestMatcher matcher(try_gpu, match_conf);
    matcher(features, pairwise_matches);
    matcher.collectGarbage();
    
    HomographyBasedEstimator estimator;
    estimator(features, pairwise_matches, cameras);
    
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
    }
    
    Ptr<detail::BundleAdjusterBase> adjuster;
    if (ba_cost_func == "reproj") adjuster = new detail::BundleAdjusterReproj();
    else if (ba_cost_func == "ray") adjuster = new detail::BundleAdjusterRay();
    else
    {
        cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
    }
    adjuster->setConfThresh(conf_thresh);
    Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
    if (ba_refine_mask[0] == 'x') refine_mask(0,0) = 1;
    if (ba_refine_mask[1] == 'x') refine_mask(0,1) = 1;
    if (ba_refine_mask[2] == 'x') refine_mask(0,2) = 1;
    if (ba_refine_mask[3] == 'x') refine_mask(1,1) = 1;
    if (ba_refine_mask[4] == 'x') refine_mask(1,2) = 1;
    adjuster->setRefinementMask(refine_mask);
    (*adjuster)(features, pairwise_matches, cameras);
}