#include <iostream>
#include <opencv2/opencv.hpp>

std::vector<int> dfs(int node,
                     std::vector< std::vector<int> >& edges,
                     std::vector<bool>& visited) {
    std::vector<int> answer;
    answer.push_back(node);
    for (int i = 0; i < edges[node].size(); ++i) {
        if(visited[edges[node][i]] == false){
            visited[edges[node][i]] = true;
            std::vector<int> subans = dfs(edges[node][i], edges, visited);
            for (int j = 0; j < subans.size(); ++j) {
                answer.push_back(subans[j]);
            }
        }
    }
    return answer;
}

std::vector< std::vector<cv::Point2f> > findClusters (std::vector<cv::Point2f> points,
                                                      double threshold) {
    std::vector< std::vector<int> > edges(points.size());
    std::vector<bool> visited(points.size(), false);

    cv::Point2f p;
    for (int i = 0; i < points.size(); ++i) {
        for (int j = i; j < points.size(); ++j) {
            p = points[i] - points[j];
            if (p.dot(p) < threshold * threshold) {
                // add edge
                edges[i].push_back(j);
            }
        }
    }

    std::vector< std::vector<cv::Point2f> > answers;

    //scan
    for (int i = 0; i < visited.size(); ++i) {
        if (visited[i] == false) {
            std::vector<cv::Point2f> answer;
            std::vector<int> ans = dfs(i, edges, visited);
            for (int j = 0; j < ans.size(); ++j) {
                answer.push_back(points[ans[j]]);
            }
            answers.push_back(answer);
        }
    }

    return answers;
}

void playVideo (cv::VideoCapture& video) {
    cv::Mat frame;

    while (cv::waitKey(33) < 0) {
        if (video.read(frame) == false) break;
        cv::imshow("video", frame);
    }
}

void playHarrisCorner (cv::VideoCapture& video) {
    cv::Mat frame, fmono, fharris, fhn, fhnabs;

    while (cv::waitKey(33) < 0) {
        if (video.read(frame) == false) break;
        cv::cvtColor(frame, fmono, cv::COLOR_BGR2GRAY);
        cv::cornerHarris(fmono, fharris, 2, 3, 0.4);
        cv::normalize(fharris, fhn, 0, 255, cv::NORM_MINMAX, CV_32FC1);
        cv::convertScaleAbs(fhn, fhnabs);

        for (int j = 0; j < fhnabs.rows; ++j) {
            for (int i = 0; i < fhnabs.cols; ++i) {
                if (fhn.at<float>(j,i) < 200) {
                    cv::circle(frame, cv::Point(i, j), 3, cv::Scalar(0, 0, 255));
                }
            }
        }
        cv::imshow("video", frame);
    }

    cv::waitKey(0);
}


void playFASTCorner (cv::VideoCapture& video) {
    cv::Mat frame, fmono;

    while (cv::waitKey(33) < 0) {
        if (video.read(frame) == false) break;
        cv::cvtColor(frame, fmono, cv::COLOR_BGR2GRAY);
        std::vector<cv::KeyPoint> fastKeyPoints;
        cv::FAST(fmono, fastKeyPoints, 17, true);
        for (unsigned i = 0; i < fastKeyPoints.size(); ++i) {
            cv::circle(frame, cv::Point(fastKeyPoints[i].pt.x, fastKeyPoints[i].pt.y), 3, cv::Scalar(0, 0, 255));
        }
        std::cout << fastKeyPoints.size() << std::endl;
        cv::imshow("video", frame);
    }

    cv::waitKey(0);
}


void playOptFlowLK (cv::VideoCapture& video, bool subtract = false) {
    cv::Mat frame1, frame2,fmono1, fmono2, frameBase, frameDiff;
    int count = 0;

    video.read(frame1);
    frameBase = frame1.clone();
    for (video.read(frame1); cv::waitKey(33) < 0; frame1 = frame2.clone()) {
        if (video.read(frame2) == false) break;

        cv::cvtColor(frame1, fmono1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(frame2, fmono2, cv::COLOR_BGR2GRAY);

        std::vector<cv::KeyPoint> fastKeyPoints;
        std::vector<cv::Point2f> points, nextpoints;
        cv::FAST(fmono1, fastKeyPoints, 17, true);
        cv::KeyPoint::convert(fastKeyPoints, points);

        std::cerr << "Count: " << ++count << std::endl;

        std::vector<unsigned char>  status;
        std::vector<float> errors;

        cv::calcOpticalFlowPyrLK(fmono1, fmono2, points, nextpoints, status, errors);

        if (subtract) {
            std::vector<cv::Point2f> substractedPoints;
            cv::Mat maskBase, mask;
            cv::absdiff(frame1, frameBase, frameDiff);
            cv::cvtColor(frameDiff, maskBase, cv::COLOR_BGR2GRAY);
            mask = cv::Mat::zeros(frameDiff.rows, frameDiff.cols, CV_8UC3);
            cv::medianBlur(maskBase, maskBase, 9);

            for (int i = 0; i < status.size(); ++i) {
                if (status[i] == 1) {
                    cv::Point pt = cv::Point(points[i].x, points[i].y);
                    cv::Point pt2 = cv::Point(nextpoints[i].x, nextpoints[i].y) - pt;
                    float l2 = pt2.dot(pt2);

                    float athres = 4.0, bthres = 50.0;
                    float a = 1.0 / athres, b = 1.0 / bthres;
                    float mag = sqrt(l2);
                    float diff = (float)(maskBase.at<char>(pt));

                    if (a * mag + b * diff > 1.0)
                        substractedPoints.push_back(pt);
                }
            }

            std::vector< std::vector<cv::Point2f> > finalPointSets = findClusters(substractedPoints, 50.0);
            std::vector<cv::Point2f> finalPoints;

            for (int i = 0; i < finalPointSets.size(); ++i) {
                if (finalPointSets[i].size() > finalPoints.size())
                    finalPoints = finalPointSets[i];
                for (int j = 0; j < finalPointSets[i].size(); ++j) {
                    cv::circle(mask, finalPointSets[i][j], 3, cv::Scalar(255, 0, 0));
                }
            }

            for (int i = 0; i < finalPoints.size(); ++i) {
                cv::circle(mask, finalPoints[i], 3, cv::Scalar(0, 0, 255));
            }
            cv::Mat bpoints = cv::Mat(finalPoints);
            cv::imshow("mask", mask);
            cv::imshow("maskbase", maskBase);

            if(bpoints.rows > 0) {
                cv::boundingRect(bpoints);
                cv::Rect bRect = cv::boundingRect(bpoints);
                cv::rectangle(frame1, bRect, cv::Scalar(0, 0, 255));
            }
        }

//        for(int i = 0; i < status.size(); ++i) {
//            if (status[i] == 1) {
//                cv::line(frame1,
//                         cv::Point(points[i].x, points[i].y),
//                         cv::Point(nextpoints[i].x, nextpoints[i].y),
//                         cv::Scalar(0, 0, 255));
//            }
//        }

        cv::imshow("video", frame1);
    }
    cv::waitKey(0);
}


void playOptFlowSF (cv::VideoCapture& video, bool subtract = false) {
    cv::Mat frame1, frame2, flow, frameBase, frameDiff;
    int count = 0;

    video.read(frame1);
    frameBase = frame1.clone();
    for (/*video.read(frame1)*/; cv::waitKey(33) < 0; frame1 = frame2.clone()) {
        if (video.read(frame2) == false) break;

        std::cerr << "Count: " << ++count << std::endl;
        if (count < 60) continue;

        std::cerr << "Calculating..." << std::endl;

        cv::calcOpticalFlowSF(frame1, frame2, flow, 3, 2, 4);
        cv::Mat xy[2];
        cv::split(flow, xy);

        std::cerr << "Calculation finished." << std::endl;

        //calculate angle and magnitude
        cv::Mat magnitude_, magnitude, angle;
        cv::cartToPolar(xy[0], xy[1], magnitude_, angle, true);

        //translate magnitude to range [0;1]
        cv::normalize(magnitude_, magnitude, 0.0, 1.0, cv::NORM_MINMAX);

        //build hsv image
        cv::Mat _hsv[3], hsv;
        _hsv[0] = angle;
        _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
        _hsv[2] = magnitude;
        merge(_hsv, 3, hsv);

        //convert to BGR and show
        cv::Mat bgr;//CV_32FC3 matrix
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

        if (subtract) {
            cv::Mat mask;
            cv::absdiff(frame1, frameBase, frameDiff);
            cv::cvtColor(frameDiff, mask, cv::COLOR_BGR2GRAY);

            for (int j = 0; j < mask.rows; ++j) {
                for (int i = 0; i < mask.cols; ++i) {
                    float athres = 1.5, bthres = 50.0;
                    float a = 1.0 / athres, b = 1.0 / bthres;
                    float mag = magnitude_.at<float>(j, i);
                    float diff = (float)(mask.at<char>(j,i));
                    if (a * mag + b * diff < 1.0)
                        mask.at<char>(j, i) = 255;
                    else
                        mask.at<char>(j, i) = 0;
                }
            }

            cv::medianBlur(mask, mask, 9);

            cv::imshow("mask", mask);
            frame1.setTo(0, mask);
        }

        cv::imshow("video", bgr);
        cv::imshow("original", frame1);
    }

    cv::waitKey(0);
}


void playOptFlow (cv::VideoCapture& video, bool subtract = false) {
    cv::Mat frame1, frame2,fmono1, fmono2, flow, frameBase, frameDiff;
    int count = 0;

    video.read(frame1);
    frameBase = frame1.clone();
    for (/*video.read(frame1)*/; cv::waitKey(33) < 0; frame1 = frame2.clone()) {
        if (video.read(frame2) == false) break;

        cv::cvtColor(frame1, fmono1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(frame2, fmono2, cv::COLOR_BGR2GRAY);

        std::cerr << "Count: " << ++count << std::endl;

        cv::calcOpticalFlowFarneback(fmono1, fmono2, flow, 0.5, 3, 15, 1, 5, 1.1, 0);
        cv::Mat xy[2];
        cv::split(flow, xy);

        //calculate angle and magnitude
        cv::Mat magnitude_, magnitude, angle;
        cv::cartToPolar(xy[0], xy[1], magnitude_, angle, true);

        //translate magnitude to range [0;1]
        cv::normalize(magnitude_, magnitude, 0.0, 1.0, cv::NORM_MINMAX);

        //build hsv image
        cv::Mat _hsv[3], hsv;
        _hsv[0] = angle;
        _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
        _hsv[2] = magnitude;
        merge(_hsv, 3, hsv);

        //convert to BGR and show
        cv::Mat bgr;//CV_32FC3 matrix
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

        if (subtract) {
            cv::Mat mask;
            cv::absdiff(frame1, frameBase, frameDiff);
            cv::cvtColor(frameDiff, mask, cv::COLOR_BGR2GRAY);

            for (int j = 0; j < mask.rows; ++j) {
                for (int i = 0; i < mask.cols; ++i) {
                    float athres = 1.5, bthres = 50.0;
                    float a = 1.0 / athres, b = 1.0 / bthres;
                    float mag = magnitude_.at<float>(j, i);
                    float diff = (float)(mask.at<char>(j,i));
                    if (a * mag + b * diff < 1.0)
                        mask.at<char>(j, i) = 255;
                    else
                        mask.at<char>(j, i) = 0;
                }
            }

            cv::medianBlur(mask, mask, 9);

            cv::imshow("mask", mask);
            frame1.setTo(0, mask);
        }

        cv::imshow("video", bgr);
        cv::imshow("original", frame1);
    }

    cv::waitKey(0);
}


void playSimpleBgSubtract (cv::VideoCapture& video) {
    cv::Mat frameBase, frame, frameDiff, mask;
    int count = 0;

    video.read(frame);
    frameBase = frame.clone();
    for (; cv::waitKey(33) < 0;) {
        if (video.read(frame) == false) break;

        std::cerr << "Count: " << ++count << std::endl;

        cv::absdiff(frame, frameBase, frameDiff);
        cv::cvtColor(frameDiff, mask, cv::COLOR_BGR2GRAY);


        for (int j = 0; j < mask.rows; ++j) {
            for (int i = 0; i < mask.cols; ++i) {
                if (mask.at<char>(j, i) < 10)
                    mask.at<char>(j, i) = 255;
                else
                    mask.at<char>(j, i) = 0;
            }
        }

//            cv::imshow("mask", mask);
//            frame1.setTo(0, mask);
//        }

        cv::imshow("video", frame);
        cv::imshow("diff", frameDiff);
        cv::imshow("mask", mask);
        cv::imshow("base", frameBase);
    }

    cv::waitKey(0);
}

int main (int argc, char* argv[]) {

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " filename command arg1 arg2..." << std::endl;
        return 0;
    }

    cv::VideoCapture video(argv[1]);
    if (!video.isOpened()) {
        std::cerr << "Error: " << "Cannot open the file." << std::endl;
        return -1;
    }

    // Command Control
    if (strcmp(argv[2], "play") == 0) {
        playVideo (video);
    } else if (strcmp(argv[2], "harris") == 0) {
        playHarrisCorner (video);
    } else if (strcmp(argv[2], "fast") == 0) {
        playFASTCorner (video);
    } else if (strcmp(argv[2], "flow") == 0) {
        playOptFlow(video);
    } else if (strcmp(argv[2], "flowsub") == 0) {
        playOptFlow(video, true);
    } else if (strcmp(argv[2], "flowsf") == 0) {
        playOptFlowSF(video);
    } else if (strcmp(argv[2], "flowsfsub") == 0) {
        playOptFlowSF(video, true);
    } else if (strcmp(argv[2], "flowlk") == 0) {
        playOptFlowLK(video);
    } else if (strcmp(argv[2], "flowlksub") == 0) {
        playOptFlowLK(video, true);
    } else if (strcmp(argv[2], "simplesub") == 0) {
        playSimpleBgSubtract(video);
    }

    std::cout << "Finished!" << std::endl;
    return 0;
}
