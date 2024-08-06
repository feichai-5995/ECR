// one-to-many PnP-RANSAC implementation for SRC.
/* 
PnP-RANSAC implementation based on DSAC++
Code: https://github.com/vislearn/LessMore
Paper: https://arxiv.org/abs/1711.10228
*/  

/*
Copyright (c) 2016, TU Dresden
Copyright (c) 2017, Heidelberg University
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the TU Dresden, Heidelberg University nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TU DRESDEN OR HEIDELBERG UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "pnpransac.h"
#include <fstream> // debug inliers.
#include <opencv2/calib3d.hpp> // try pose refinement.

# define CV_ITERATIVE 0
# define CV_P3P 2

std::vector<std::mt19937> ThreadRand::generators;
bool ThreadRand::initialised = false;

void ThreadRand::forceInit(unsigned seed)
{
    initialised = false;
    init(seed);
}

void ThreadRand::init(unsigned seed)
{
    #pragma omp critical
    {
	if(!initialised)
	{
	    unsigned nThreads = omp_get_max_threads();
	    
	    for(unsigned i = 0; i < nThreads; i++)
	    {    
		generators.push_back(std::mt19937());
		generators[i].seed(i+seed);
	    }

	    initialised = true;
	}    
    }
}

int ThreadRand::irand(int min, int max, int tid)
{
    std::uniform_int_distribution<int> dist(min, max);

    unsigned threadID = omp_get_thread_num();
    if(tid >= 0) threadID = tid;
    
    if(!initialised) init();
  
    return dist(ThreadRand::generators[threadID]);
}

double ThreadRand::drand(double min, double max, int tid)
{
    std::uniform_real_distribution<double> dist(min, max);
    
    unsigned threadID = omp_get_thread_num();
    if(tid >= 0) threadID = tid;

    if(!initialised) init();

    return dist(ThreadRand::generators[threadID]);
}

double ThreadRand::dgauss(double mean, double stdDev, int tid)
{
    std::normal_distribution<double> dist(mean, stdDev);
    
    unsigned threadID = omp_get_thread_num();
    if(tid >= 0) threadID = tid;

    if(!initialised) init();

    return dist(ThreadRand::generators[threadID]);
}

int irand(int incMin, int excMax, int tid)
{
    return ThreadRand::irand(incMin, excMax - 1, tid);
}

double drand(double incMin, double incMax,int tid)
{
    return ThreadRand::drand(incMin, incMax, tid);
}

int igauss(int mean, int stdDev, int tid)
{
    return (int) ThreadRand::dgauss(mean, stdDev, tid);
}

double dgauss(double mean, double stdDev, int tid)
{
    return ThreadRand::dgauss(mean, stdDev, tid);
}

namespace poseSolver {
    
    std::pair<cv::Mat, cv::Mat> getInvHyp(const std::pair<cv::Mat, cv::Mat>& hyp)
    {
        cv::Mat_<double> hypR, trans = cv::Mat_<float>::eye(4, 4);
        cv::Rodrigues(hyp.first, hypR);

        hypR.copyTo(trans.rowRange(0,3).colRange(0,3));
        trans(0, 3) = hyp.second.at<double>(0, 0);
        trans(1, 3) = hyp.second.at<double>(0, 1);
        trans(2, 3) = hyp.second.at<double>(0, 2);

        trans = trans.inv();

        std::pair<cv::Mat, cv::Mat> invHyp;
        cv::Rodrigues(trans.rowRange(0,3).colRange(0,3), invHyp.first);
        invHyp.second = cv::Mat_<double>(1, 3);
        invHyp.second.at<double>(0, 0) = trans(0, 3);
        invHyp.second.at<double>(0, 1) = trans(1, 3);
        invHyp.second.at<double>(0, 2) = trans(2, 3);

        return invHyp;
    }

    double calcAngularDistance(const std::pair<cv::Mat, cv::Mat> & h1, const std::pair<cv::Mat, cv::Mat> & h2)
    {
        cv::Mat r1, r2;
        cv::Rodrigues(h1.first, r1);
        cv::Rodrigues(h2.first, r2);

        cv::Mat rotDiff= r2 * r1.t();
        double trace = cv::trace(rotDiff)[0];

        trace = std::min(3.0, std::max(-1.0, trace));
        return 180*acos((trace-1.0)/2.0)/CV_PI;
    }

    double maxLoss(const std::pair<cv::Mat, cv::Mat> & h1, const std::pair<cv::Mat, cv::Mat> & h2)
    {
        // measure loss of inverted poses (camera pose instead of scene pose)
        std::pair<cv::Mat, cv::Mat>  invH1 = getInvHyp(h1);
        std::pair<cv::Mat, cv::Mat>  invH2 = getInvHyp(h2);

        double rotErr = calcAngularDistance(invH1, invH2);
        double tErr = cv::norm(invH1.second - invH2.second);

        return std::max(rotErr, tErr * 100);
    }

    inline bool safeSolvePnP(
        // 再次重建，cv::solvePnP执行时如果rot, trans不为0则会进行优化
        const std::vector<cv::Point3f>& objPts,
        const std::vector<cv::Point2f>& imgPts,
        const cv::Mat& camMat,
        const cv::Mat& distCoeffs,
        cv::Mat& rot,
        cv::Mat& trans,
        bool extrinsicGuess,
        int methodFlag)
    {
        if(rot.type() == 0) rot = cv::Mat_<double>::zeros(1, 3);
        if(trans.type() == 0) trans= cv::Mat_<double>::zeros(1, 3);

        if(!cv::solvePnP(objPts, imgPts, camMat, distCoeffs, rot, trans, extrinsicGuess, methodFlag))
        {
            rot = cv::Mat_<double>::zeros(1, 3);
            trans = cv::Mat_<double>::zeros(1, 3);
            return false;
        }
        return true;
    }

    PnPRANSAC::PnPRANSAC () {
        this->camMat = cv::Mat_<float>::eye(3, 3);
    }
    
    PnPRANSAC::PnPRANSAC (float fx, float fy, float cx, float cy) {
        this->camMat = cv::Mat_<float>::eye(3, 3);
        this->camMat(0,0) = fx;
        this->camMat(1,1) = fy;
        this->camMat(0,2) = cx;
        this->camMat(1,2) = cy;
    }
    
    PnPRANSAC::~PnPRANSAC () {}
    
    void PnPRANSAC::camMatUpdate(float fx, float fy, float cx, float cy){
        this->camMat = cv::Mat_<float>::eye(3, 3);
        this->camMat(0,0) = fx;
        this->camMat(1,1) = fy;
        this->camMat(0,2) = cx;
        this->camMat(1,2) = cy;
    }


/*
python调用该函数时输入的参数和这里的c++对应不上是因为pyhind还有个中间过渡的函数
    def RANSAC_one2many(self, 
                 np.ndarray[double, ndim=2, mode="c"] img_pts, 
                 np.ndarray[double, ndim=3, mode="c"] obj_pts, 
                 int n_hyp):
        #print(obj_pts[66][3])
        cdef float[:, :] img_pts_ = img_pts.astype(np.float32)
        cdef float[:, :, :] obj_pts_ = obj_pts.astype(np.float32)
        cdef int n_pts, n_cps
        n_pts, n_cps = img_pts_.shape[0], obj_pts_.shape[1]
        assert img_pts_.shape[0] == obj_pts_.shape[0]
        cdef double* pose
        pose = self.c_pnpransac.RANSAC_one2many(&img_pts_[0,0], &obj_pts_[0,0,0], n_pts, n_cps, n_hyp)
        rot =  np.array([pose[0],pose[1],pose[2]])
        transl = np.array([pose[3],pose[4],pose[5]])
        return rot, transl
*/

    // 总体重建逻辑是这么回事，随机抽取点，抽取出四个不重复像素点重建，随机数包含点序号和多图像的图像序号
    // 重建结果要求映射的2D坐标与实际差距小于阈值，然后逐像素记录2D偏差，多张图像就记录偏差最小的那个
    // 这样子捣鼓出256组重建结果，然后选择分数最低的那组，迭代200次重建优化，
    // 迭代方式是带入所有2D偏差小于阈值的像素到CV_ITERATIVE方案，配合原本的旋转向量和平移向量计算
    // 当迭代无法在优化到某阈值之上时停止迭代优化，返回旋转向量和平移向量
    double* PnPRANSAC::RANSAC_one2many(
        // 函数有很多可以修改的地方，特别是当前针对多token图像输入的情况，该情况下有很多计算量冗余
        // 那么函数改良有了，创新主要是简化和效率，重点就在于如何到达之前的效果，直觉告诉我关键在阈值
        float* imgPts_,
        // 理论上的二维点坐标
        float* objPts_,
        int nPts,
        int nCps,
        // 这两个参数分别为前两个参数的shape0和shape1，也就是像素点数和图像数
        int objHyps,
        // 输入的超参数256
        int flag=0)
    {
        // hyper params.
        int inlierThreshold2D = 10; // 10 default.
        // 这里的阈值用于得分计算，不能瞎调
        int refSteps = 100;
        // format data.
        std::vector<cv::Point2f> imgPts(nPts);
        std::vector<std::vector<cv::Point3f>> objPts(nPts);
        // 构建等大于像素点数的std::vector动态数组
        for (unsigned i = 0; i < objPts.size(); i++)
        // 像素点数次循环
            objPts[i] = std::vector<cv::Point3f>(nCps);
            // 构建数量大小等于像素数的数组
        #pragma omp parallel for
        // 同理于dsac*当中的语句，openmp当中并行的语句
        for(unsigned i=0; i<imgPts.size(); i++)
            imgPts[i] = cv::Point2f(imgPts_[i*2], imgPts_[i*2+1]); // default.
            // 存储二维点
            //imgPts[i] = cv::Point2f(imgPts_[i*2]+4, imgPts_[i*2+1]+4); // shift.
        #pragma omp parallel for
        for(unsigned i=0; i<objPts.size(); i++)
            for (unsigned j = 0; j < nCps; j++)
                objPts[i][j] = cv::Point3f(objPts_[i*nCps*3+j*3], objPts_[i*nCps*3+j*3 + 1], objPts_[i*nCps*3+j*3 + 2]);
                // 提取三维坐标信息，应该是预处理当中使用了指针压缩了三维列表
                // objPts的第一个维度是所有像素，第二个维度是所有图像数
                // nCps针对的是多张图像
        // variables.

        std::vector<std::vector<cv::Point2f>> sampledImgPts(objHyps);
        std::vector<std::vector<cv::Point3f>> sampledObjPts(objHyps);
        std::vector<cv::Mat_<double>> rotHyp(objHyps);
        std::vector<cv::Mat_<double>> tHyp(objHyps);
        std::vector<float> scores(objHyps);
        std::vector<std::vector<float>> reproDiff(objHyps);
        std::vector<std::vector<int>> correspID(objHyps);
        // sample hypotheses.
        #pragma omp parallel for
        for(int h = 0; h < objHyps; h++)
        // 循环256次，也就是找出256组，太慢后面需要修改，dsac*使用64组
        while(true)
        {
            std::vector<cv::Point2f> projections;
            std::vector<int> alreadyChosen(nPts,0);
            // 创建整形张量，长度nPts对应像素点数量，填充为0
            sampledImgPts[h].clear();
            sampledObjPts[h].clear();
            for(int j = 0; j < 4; j++)
            // 还是四个点的定位
            // 所以这瓜皮的逻辑就是随机数生成，两次被随机到就作为重建点
            {
                int idx = irand(0, nPts);
                // 指向ThreadRand::irand(incMin, excMax - 1, tid)，多线程随机数生成函数
                // 每个点的序号
                // 这里和dsac*处理方式就不一样了，dsac当中保留二维
                if(alreadyChosen[idx] > 0)
                // 该点已被选中
                {
                    j--;
                    continue;
                }
                int cpIdx = irand(0, nCps);
                // nCps，也就是兼容于带入多张图像，从多张图像当中随机抽取像素
                // 所以这里得到图像序号
                alreadyChosen[idx] = 1;
                // 选中该点
                sampledImgPts[h].push_back(imgPts[idx]); // 2D coordinate.
                sampledObjPts[h].push_back(objPts[idx][cpIdx]); // 3D coordinate.
                // 记录该点的2D坐标和3D坐标
            }
            if(!safeSolvePnP(sampledObjPts[h], sampledImgPts[h], 
                this->camMat, cv::Mat(), rotHyp[h], tHyp[h], false, CV_P3P)) continue;
            // 四点求解，求解不出就continue

            // check reconstruction, 4 sampled points should be reconstructed perfectly.
            // 检查重建，4个采样点应完全重建。
            cv::projectPoints(sampledObjPts[h], 
                rotHyp[h], tHyp[h], this->camMat, cv::Mat(), projections);
                // 第一个参数是输入的3D点坐标，最后一个参数是映射的2D点
            
            bool foundOutlier = false;
            for(unsigned j = 0; j < sampledImgPts[h].size(); j++)
            // 遍历抽取的四个点的3D坐标
            {
                if(cv::norm(sampledImgPts[h][j] - projections[j]) < inlierThreshold2D)
                    continue;
                    // 输入的二维平面坐标和重建完3D映射出的二维平面坐标差距小于阈值
                foundOutlier = true;
                break;
            }
            if(foundOutlier)
                continue;
                // 重新映射的二维点偏差太大则重置
            else{
                // compute reprojection error and hypothesis score.
                // 计算重投影误差和假设得分。
                std::vector<cv::Point2f> projections; 
                std::vector<cv::Point3f> cpPts(nPts*nCps);
                for (int i = 0; i < nPts; ++i)
                    for (int j = 0; j < nCps; ++j)
                        cpPts[i*nCps+j] = objPts[i][j];
                        // 将所有三维像素转存，针对同一像素的多张图像结果会被放一起
                cv::projectPoints(cpPts, rotHyp[h], tHyp[h], this->camMat, cv::Mat(), projections);
                // 三维点映射到二维，带入了所有token图像
                std::vector<float> diff(nPts);
                std::vector<int> cpID(nPts);
                float score = 0.;
                // #pragma omp parallel for
                for(unsigned pt = 0; pt < imgPts.size(); pt++)
                // 2D像素数次循环，也就是逐像素遍历
                {
                    float err = 999.;
                    // min err among correspondences.
                    for (unsigned cp = 0; cp < nCps; cp++)
                    // 图像数次循环，每张图像的每个像素计算2D偏差
                    {
                        float residual = cv::norm(imgPts[pt] - projections[pt*nCps+cp]);
                        if (residual < err) 
                        {
                            err = residual;
                            cpID[pt] = cp;
                        }
                    }
                    // 所以找出了所有图像重建结果中每个像素误差最小的

                    if (flag!=0)
                    // 默认为0，外部按照常理能输入和改变但是python接口将其覆盖
                    {
                        // mean err among correspondences.
                        err = 0.;
                        for (unsigned cp = 0; cp < nCps; cp++)
                        {
                            float residual = cv::norm(imgPts[pt] - projections[pt*nCps+cp]);
                            err += residual;
                        }
                        err /= nCps;                        
                    }
                    diff[pt] = err;
                    // compute the score.
                    score = score + (1. / (1. + std::exp(-(0.5*(err-inlierThreshold2D)))));
                    // 得分的计算方式还挺复杂……至少我打不出来最好画个公式
                }
                reproDiff[h] = diff;
                correspID[h] = cpID;
                // 存储重映射到2D的所有带入token图像中误差最小的每个像素
                scores[h] = score;
                break;
                // 这个break跳出的是while true
                // 每次while ture创建256次重建结果之一
            }
        }
        // select one winning hypothesis.
        // 选择一个获胜的假设。
        // std::min_element查找的是序列中最小的元素，这里返回的是序号
        int hypIdx = std::min_element(scores.begin(),scores.end()) - scores.begin(); 

        // refine the hypothesis.
        // 完善假设。
        double convergenceThresh = 0.01; // 0.01 by default.
        std::vector<float> localDiff = reproDiff[hypIdx];
        std::vector<int> localCpID = correspID[hypIdx];
        // 得到当前这个得分最低的旋转向量和平移向量的diff和cpid，分别对应所有token像素当中映射到2D相比于之前偏移误差最小值和序号


        // 超参数200循环，重定位两百次，如果偏离小于阈值的像素数小于4或者无法重建都会break，重建改进效果小于阈值也会跳出重建
        // 目标是多次重建提高精度么
        // 有效的像素数基本不会减少，所以实际上重建方案是CV_ITERATIVE，也就是多个像素点同时使用
        for(int rStep = 0; rStep < refSteps; rStep++)
        {
            // collect inliers.
            std::vector<cv::Point2f> localImgPts;
            std::vector<cv::Point3f> localObjPts;
            for(int pt = 0; pt < nPts; pt++)
            // 像素点数次循环
            // 转存了所有2D偏离小于阈值的像素
            {
                if(localDiff[pt] < inlierThreshold2D)
                // 2D像素偏移距离 
                {
                    localImgPts.push_back(imgPts[pt]);
                    localObjPts.push_back(objPts[pt][localCpID[pt]]);
                }
            }
            if(localImgPts.size() < 4)
                break;
            // 转成2D后误差范围之内的像素少于4个 
            // recalculate pose.
            cv::Mat_<double> rotNew = rotHyp[hypIdx].clone();
            cv::Mat_<double> tNew = tHyp[hypIdx].clone();
            // 获取之前重建的256组当中得分最低那组旋转向量和平移向量 
            // 写入之前重建得到的旋转向量和平移向量
            
            if(!safeSolvePnP(localObjPts, localImgPts, this->camMat, 
                    cv::Mat(), rotNew, tNew, true, (localImgPts.size() > 4) ? CV_ITERATIVE : CV_P3P))
                break; // abort if PnP fails.
                // 自定义的函数，指针形式的rotNew和tNew会被函数里面影响到
                // safeSolvePnP会以新获取到的点对rotNew, tNew进行优化重建，但如果求解不出会被置零
                
                // (localImgPts.size() > 4) ? CV_ITERATIVE : CV_P3P
                // 这玩意算一段，如果有效点数大于4执行第一种算法，
                // 所以和dsac相比或许算法也不一样了，只有刚好四个点才会CV_P3P
                // 这种情况下，要么降低阈值减少符合的点数，不然点数上升确实可能计算被带飞

            std::pair<cv::Mat, cv::Mat> hypNew;
            std::pair<cv::Mat, cv::Mat> hypOld;
            hypNew.first = rotNew;
            hypNew.second = tNew;
            // 获取的新旋转向量和平移向量
            hypOld.first = rotHyp[hypIdx];
            hypOld.second = tHyp[hypIdx];
            if(maxLoss(hypNew, hypOld) < convergenceThresh) // check convergence.
                break;
            // 计算出的新旋转向量和平移向量于之前的向量差距小于阈值则跳出
            // 所以这里重建的目的是返回偏差足够小的新平移向量和旋转矩阵
            // 换而言之，采用新点对旋转向量和平移向量进行优化，如果优化效果小于阈值则提前跳出
            rotHyp[hypIdx] = rotNew;
            tHyp[hypIdx] = tNew;
            // 更新优化结果

            // update correspondences.
            // 更新对应关系
            std::vector<cv::Point2f> projections;
            std::vector<cv::Point3f> cpPts(nPts*nCps);
            for (int i = 0; i < nPts; ++i)
                for (int j = 0; j < nCps; ++j)
                // 像素点数和token图像数的遍历
                    cpPts[i*nCps+j] = objPts[i][j];
            // 转存每张token图像的每个3D像素
            cv::projectPoints(cpPts, rotHyp[hypIdx], tHyp[hypIdx], this->camMat, cv::Mat(), projections);
            // 三维点映射到二维，映射了token当中的每张图
            // 根据新的相机位姿优化结果对像素进行映射，以便后续更新出新的2D偏差最小值和所在点
            std::vector<float> diff(nPts);
            std::vector<int> cpID(nPts);

        // token所有图像，每个像素投影到2D，与实际2D误差最小的误差和ID分别记录在diff和cpID   
            #pragma omp parallel for
            for(unsigned pt = 0; pt < imgPts.size(); pt++)
            // 逐像素遍历
            {
                //if (rStep < 100)
                if (true)
                {
                    float err = 999.;
                    // min err among correspondences.
                    for (unsigned cp = 0; cp < nCps; cp++)
                    // 遍历每张图
                    {
                        float residual = cv::norm(imgPts[pt] - projections[pt*nCps+cp]);
                        // 输入的二维坐标与映射的二维坐标的差值
                        if (residual < err) 
                        {
                            err = residual;
                            cpID[pt] = cp;
                        }
                    }
                    // 迭代出所有token图，映射到2D后最小的差值和序号
                    if (flag!=0) // mean err among correspondences.
                    {
                        err = 0.;
                        for (unsigned cp = 0; cp < nCps; cp++)
                        {
                            float residual = cv::norm(imgPts[pt] - projections[pt*nCps+cp]);
                            err += residual;
                        }
                        err /= nCps;                        
                    }
                    diff[pt] = err;
                    // 所有token当中针对每个像素最小的err存储数组
                }
            }
            localDiff = diff;
            localCpID = cpID;
        }

/*
        // // debug: save the inliers to file.
        // std::vector<cv::Point2f> inliers2D;
        // std::vector<cv::Point3f> inliers3D;
        // for(int pt = 0; pt < nPts; pt++)
        // {
        //     if(localDiff[pt] < inlierThreshold2D)
        //     {
        //         inliers2D.push_back(imgPts[pt]);
        //         inliers3D.push_back(objPts[pt][localCpID[pt]]);
        //     }
        // }
        // std::ofstream file2D, file3D;
        // file2D.open("_inliers2D.txt");
        // file3D.open("_inliers3D.txt");
        // for (int idx=0; idx<inliers2D.size(); idx++)
        // {
        //     file2D << inliers2D[idx].x << " " << inliers2D[idx].y << "\n";
        //     file3D << inliers3D[idx].x << " " << inliers3D[idx].y << " " << inliers3D[idx].z << "\n";
        // }
        // file2D.close();
        // file3D.close();
*/


        static double  pose[6];
        for (int i = 0; i < 3; i++) 
            pose[i] = rotHyp[hypIdx](0,i);
        for (int i = 3; i < 6; i++) 
            pose[i] = tHyp[hypIdx](0,i-3);
        return pose;
        // 输出六位，前三位是旋转向量后三位是平移向量
    }
}

