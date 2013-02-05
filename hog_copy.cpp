#include "precomp.hpp"
#include <iterator>
#ifdef HAVE_IPP
#include "ipp.h"
#endif

using namespace std;

namespace cv{

struct myHOGCache
{
    struct BlockData
    {
        BlockData() : histOfs(0), imgOffset() {}
        int histOfs;
        Point imgOffset;
    };

    struct PixData
    {
        size_t gradOfs, qangleOfs;
        int histOfs[4];
        float histWeights[4];
        float gradWeight;
    };

    myHOGCache();
    myHOGCache(const myHOGDescriptor* descriptor,
        const Mat& img, Size paddingTL, Size paddingBR,
        bool useCache, Size cacheStride);
    virtual ~myHOGCache() {};
    virtual void init(const myHOGDescriptor* descriptor,
        const Mat& img, Size paddingTL, Size paddingBR,
        bool useCache, Size cacheStride);

    Size windowsInImage(Size imageSize, Size winStride) const;
    Rect getWindow(Size imageSize, Size winStride, int idx) const;

    const float* getBlock(Point pt, float* buf);
    virtual void normalizeBlockHistogram(float* histogram) const;

    vector<PixData> pixData;
    vector<BlockData> blockData;

    bool useCache;
    vector<int> ymaxCached;
    Size winSize, cacheStride;
    Size nblocks, ncells;
    int blockHistogramSize;
    int count1, count2, count4;
    Point imgoffset;
    Mat_<float> blockCache;
    Mat_<uchar> blockCacheFlags;

    Mat grad, qangle;
    const myHOGDescriptor* descriptor;
};


myHOGCache::myHOGCache()
{
    useCache = false;
    blockHistogramSize = count1 = count2 = count4 = 0;
    descriptor = 0;
}

myHOGCache::myHOGCache(const myHOGDescriptor* _descriptor,
        const Mat& _img, Size _paddingTL, Size _paddingBR,
        bool _useCache, Size _cacheStride)
{
    init(_descriptor, _img, _paddingTL, _paddingBR, _useCache, _cacheStride);
}

void myHOGCache::init(const myHOGDescriptor* _descriptor,
        const Mat& _img, Size _paddingTL, Size _paddingBR,
        bool _useCache, Size _cacheStride) // _useCache is true in this case
{
    descriptor = _descriptor;
    cacheStride = _cacheStride;
    useCache = _useCache;

    descriptor->computeGradient(_img, grad, qangle, _paddingTL, _paddingBR); // compute gradient for rescaled img
    imgoffset = _paddingTL;

    winSize = descriptor->winSize;
    Size blockSize = descriptor->blockSize;
    Size blockStride = descriptor->blockStride;
    Size cellSize = descriptor->cellSize;
    Size winSize = descriptor->winSize;
    int i, j, nbins = descriptor->nbins;
    int rawBlockSize = blockSize.width*blockSize.height; // block size

    nblocks = Size((winSize.width - blockSize.width)/blockStride.width + 1,
                   (winSize.height - blockSize.height)/blockStride.height + 1); // number of blocks, 2d vector???
    ncells = Size(blockSize.width/cellSize.width, blockSize.height/cellSize.height);
    blockHistogramSize = ncells.width*ncells.height*nbins;

    if( useCache )
    {
        Size cacheSize((grad.cols - blockSize.width)/cacheStride.width+1,
                       (winSize.height/cacheStride.height)+1);
        blockCache.create(cacheSize.height, cacheSize.width*blockHistogramSize);
        blockCacheFlags.create(cacheSize);
        size_t i, cacheRows = blockCache.rows;
        ymaxCached.resize(cacheRows);
        for( i = 0; i < cacheRows; i++ )
            ymaxCached[i] = -1;
    }

    Mat_<float> weights(blockSize);
    float sigma = (float)descriptor->getWinSigma();
    float scale = 1.f/(sigma*sigma*2);

    for(i = 0; i < blockSize.height; i++)
        for(j = 0; j < blockSize.width; j++)
        {
            float di = i - blockSize.height*0.5f;
            float dj = j - blockSize.width*0.5f;
            weights(i,j) = std::exp(-(di*di + dj*dj)*scale);
        }

    blockData.resize(nblocks.width*nblocks.height);
    pixData.resize(rawBlockSize*3);

    // Initialize 2 lookup tables, pixData & blockData.
    // Here is why:
    //
    // The detection algorithm runs in 4 nested loops (at each pyramid layer):
    //  loop over the windows within the input image
    //    loop over the blocks within each window
    //      loop over the cells within each block
    //        loop over the pixels in each cell
    //
    // As each of the loops runs over a 2-dimensional array,
    // we could get 8(!) nested loops in total, which is very-very slow.
    //
    // To speed the things up, we do the following:
    //   1. loop over windows is unrolled in the HOGDescriptor::{compute|detect} methods;
    //         inside we compute the current search window using getWindow() method.
    //         Yes, it involves some overhead (function call + couple of divisions),
    //         but it's tiny in fact.
    //   2. loop over the blocks is also unrolled. Inside we use pre-computed blockData[j]
    //         to set up gradient and histogram pointers.
    //   3. loops over cells and pixels in each cell are merged
    //       (since there is no overlap between cells, each pixel in the block is processed once)
    //      and also unrolled. Inside we use PixData[k] to access the gradient values and
    //      update the histogram
    //
    count1 = count2 = count4 = 0;
    for( j = 0; j < blockSize.width; j++ )
        for( i = 0; i < blockSize.height; i++ )
        {
            PixData* data = 0;
            float cellX = (j+0.5f)/cellSize.width - 0.5f;
            float cellY = (i+0.5f)/cellSize.height - 0.5f;
            int icellX0 = cvFloor(cellX);
            int icellY0 = cvFloor(cellY);
            int icellX1 = icellX0 + 1, icellY1 = icellY0 + 1;
            cellX -= icellX0;
            cellY -= icellY0;

            if( (unsigned)icellX0 < (unsigned)ncells.width &&
                (unsigned)icellX1 < (unsigned)ncells.width )
            {
                if( (unsigned)icellY0 < (unsigned)ncells.height &&
                    (unsigned)icellY1 < (unsigned)ncells.height )
                {
                    data = &pixData[rawBlockSize*2 + (count4++)];
                    data->histOfs[0] = (icellX0*ncells.height + icellY0)*nbins;
                    data->histWeights[0] = (1.f - cellX)*(1.f - cellY);
                    data->histOfs[1] = (icellX1*ncells.height + icellY0)*nbins;
                    data->histWeights[1] = cellX*(1.f - cellY);
                    data->histOfs[2] = (icellX0*ncells.height + icellY1)*nbins;
                    data->histWeights[2] = (1.f - cellX)*cellY;
                    data->histOfs[3] = (icellX1*ncells.height + icellY1)*nbins;
                    data->histWeights[3] = cellX*cellY;
                }
                else
                {
                    data = &pixData[rawBlockSize + (count2++)];
                    if( (unsigned)icellY0 < (unsigned)ncells.height )
                    {
                        icellY1 = icellY0;
                        cellY = 1.f - cellY;
                    }
                    data->histOfs[0] = (icellX0*ncells.height + icellY1)*nbins;
                    data->histWeights[0] = (1.f - cellX)*cellY;
                    data->histOfs[1] = (icellX1*ncells.height + icellY1)*nbins;
                    data->histWeights[1] = cellX*cellY;
                    data->histOfs[2] = data->histOfs[3] = 0;
                    data->histWeights[2] = data->histWeights[3] = 0;
                }
            }
            else
            {
                if( (unsigned)icellX0 < (unsigned)ncells.width )
                {
                    icellX1 = icellX0;
                    cellX = 1.f - cellX;
                }

                if( (unsigned)icellY0 < (unsigned)ncells.height &&
                    (unsigned)icellY1 < (unsigned)ncells.height )
                {
                    data = &pixData[rawBlockSize + (count2++)];
                    data->histOfs[0] = (icellX1*ncells.height + icellY0)*nbins;
                    data->histWeights[0] = cellX*(1.f - cellY);
                    data->histOfs[1] = (icellX1*ncells.height + icellY1)*nbins;
                    data->histWeights[1] = cellX*cellY;
                    data->histOfs[2] = data->histOfs[3] = 0;
                    data->histWeights[2] = data->histWeights[3] = 0;
                }
                else
                {
                    data = &pixData[count1++];
                    if( (unsigned)icellY0 < (unsigned)ncells.height )
                    {
                        icellY1 = icellY0;
                        cellY = 1.f - cellY;
                    }
                    data->histOfs[0] = (icellX1*ncells.height + icellY1)*nbins;
                    data->histWeights[0] = cellX*cellY;
                    data->histOfs[1] = data->histOfs[2] = data->histOfs[3] = 0;
                    data->histWeights[1] = data->histWeights[2] = data->histWeights[3] = 0;
                }
            }
            data->gradOfs = (grad.cols*i + j)*2;
            data->qangleOfs = (qangle.cols*i + j)*2;
            data->gradWeight = weights(i,j);
        }

    assert( count1 + count2 + count4 == rawBlockSize );
    // defragment pixData
    for( j = 0; j < count2; j++ )
        pixData[j + count1] = pixData[j + rawBlockSize];
    for( j = 0; j < count4; j++ )
        pixData[j + count1 + count2] = pixData[j + rawBlockSize*2];
    count2 += count1;
    count4 += count2;

    // initialize blockData
    for( j = 0; j < nblocks.width; j++ )
        for( i = 0; i < nblocks.height; i++ )
        {
            BlockData& data = blockData[j*nblocks.height + i];
            data.histOfs = (j*nblocks.height + i)*blockHistogramSize;
            data.imgOffset = Point(j*blockStride.width,i*blockStride.height);
        }
}


const float* myHOGCache::getBlock(Point pt, float* buf)
{
    float* blockHist = buf;
    assert(descriptor != 0);

    Size blockSize = descriptor->blockSize;
    pt += imgoffset;

    CV_Assert( (unsigned)pt.x <= (unsigned)(grad.cols - blockSize.width) &&
               (unsigned)pt.y <= (unsigned)(grad.rows - blockSize.height) );

    if( useCache )
    {
        CV_Assert( pt.x % cacheStride.width == 0 &&
                   pt.y % cacheStride.height == 0 );
        Point cacheIdx(pt.x/cacheStride.width,
                      (pt.y/cacheStride.height) % blockCache.rows);
        if( pt.y != ymaxCached[cacheIdx.y] )
        {
            Mat_<uchar> cacheRow = blockCacheFlags.row(cacheIdx.y);
            cacheRow = (uchar)0;
            ymaxCached[cacheIdx.y] = pt.y;
        }

        blockHist = &blockCache[cacheIdx.y][cacheIdx.x*blockHistogramSize];
        uchar& computedFlag = blockCacheFlags(cacheIdx.y, cacheIdx.x);
        if( computedFlag != 0 )
            return blockHist;
        computedFlag = (uchar)1; // set it at once, before actual computing
    }

    int k, C1 = count1, C2 = count2, C4 = count4;
    const float* gradPtr = (const float*)(grad.data + grad.step*pt.y) + pt.x*2;
    const uchar* qanglePtr = qangle.data + qangle.step*pt.y + pt.x*2;

    CV_Assert( blockHist != 0 );
#ifdef HAVE_IPP
    ippsZero_32f(blockHist,blockHistogramSize);
#else
    for( k = 0; k < blockHistogramSize; k++ )
        blockHist[k] = 0.f;
#endif

    const PixData* _pixData = &pixData[0];

    for( k = 0; k < C1; k++ )
    {
        const PixData& pk = _pixData[k];
        const float* a = gradPtr + pk.gradOfs;
        float w = pk.gradWeight*pk.histWeights[0];
        const uchar* h = qanglePtr + pk.qangleOfs;
        int h0 = h[0], h1 = h[1];
        float* hist = blockHist + pk.histOfs[0];
        float t0 = hist[h0] + a[0]*w;
        float t1 = hist[h1] + a[1]*w;
        hist[h0] = t0; hist[h1] = t1;
    }

    for( ; k < C2; k++ )
    {
        const PixData& pk = _pixData[k];
        const float* a = gradPtr + pk.gradOfs;
        float w, t0, t1, a0 = a[0], a1 = a[1];
        const uchar* h = qanglePtr + pk.qangleOfs;
        int h0 = h[0], h1 = h[1];

        float* hist = blockHist + pk.histOfs[0];
        w = pk.gradWeight*pk.histWeights[0];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;

        hist = blockHist + pk.histOfs[1];
        w = pk.gradWeight*pk.histWeights[1];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;
    }

    for( ; k < C4; k++ )
    {
        const PixData& pk = _pixData[k];
        const float* a = gradPtr + pk.gradOfs;
        float w, t0, t1, a0 = a[0], a1 = a[1];
        const uchar* h = qanglePtr + pk.qangleOfs;
        int h0 = h[0], h1 = h[1];

        float* hist = blockHist + pk.histOfs[0];
        w = pk.gradWeight*pk.histWeights[0];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;

        hist = blockHist + pk.histOfs[1];
        w = pk.gradWeight*pk.histWeights[1];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;

        hist = blockHist + pk.histOfs[2];
        w = pk.gradWeight*pk.histWeights[2];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;

        hist = blockHist + pk.histOfs[3];
        w = pk.gradWeight*pk.histWeights[3];
        t0 = hist[h0] + a0*w;
        t1 = hist[h1] + a1*w;
        hist[h0] = t0; hist[h1] = t1;
    }

    normalizeBlockHistogram(blockHist);

    return blockHist;
}


void myHOGCache::normalizeBlockHistogram(float* _hist) const
{
    float* hist = &_hist[0];
#ifdef HAVE_IPP
    size_t sz = blockHistogramSize;
#else
    size_t i, sz = blockHistogramSize;
#endif

    float sum = 0;
#ifdef HAVE_IPP
    ippsDotProd_32f(hist,hist,sz,&sum);
#else
    for( i = 0; i < sz; i++ )
        sum += hist[i]*hist[i];
#endif

    float scale = 1.f/(std::sqrt(sum)+sz*0.1f), thresh = (float)descriptor->L2HysThreshold;
#ifdef HAVE_IPP
    ippsMulC_32f_I(scale,hist,sz);
    ippsThreshold_32f_I( hist, sz, thresh, ippCmpGreater );
    ippsDotProd_32f(hist,hist,sz,&sum);
#else
    for( i = 0, sum = 0; i < sz; i++ )
    {
        hist[i] = std::min(hist[i]*scale, thresh);
        sum += hist[i]*hist[i];
    }
#endif

    scale = 1.f/(std::sqrt(sum)+1e-3f);
#ifdef HAVE_IPP
    ippsMulC_32f_I(scale,hist,sz);
#else
    for( i = 0; i < sz; i++ )
        hist[i] *= scale;
#endif
}


Size myHOGCache::windowsInImage(Size imageSize, Size winStride) const
{
    return Size((imageSize.width - winSize.width)/winStride.width + 1,
                (imageSize.height - winSize.height)/winStride.height + 1);
}

Rect myHOGCache::getWindow(Size imageSize, Size winStride, int idx) const
{
    int nwindowsX = (imageSize.width - winSize.width)/winStride.width + 1;
    int y = idx / nwindowsX;
    int x = idx - nwindowsX*y;
    return Rect( x*winStride.width, y*winStride.height, 
				winSize.width, winSize.height );
}
void myHOGDescriptor::compute(const Mat& img, vector<float>& descriptors,
                             Size winStride, Size padding,
                             const vector<Point>& locations) const
 {
     //Size()表示长和宽都是0
     if( winStride == Size() )
         winStride = cellSize;
     //gcd为求最大公约数，如果采用默认值的话，则2者相同
     Size cacheStride(gcd(winStride.width, blockStride.width),
                      gcd(winStride.height, blockStride.height));
     size_t nwindows = locations.size();
     //alignSize(m, n)返回n的倍数大于等于m的最小值
     padding.width = (int)alignSize(std::max(padding.width, 0), cacheStride.width);
     padding.height = (int)alignSize(std::max(padding.height, 0), cacheStride.height);
     Size paddedImgSize(img.cols + padding.width*2, img.rows + padding.height*2);
 
     myHOGCache cache(this, img, padding, padding, nwindows == 0, cacheStride);
 
     if( !nwindows )
         //Mat::area()表示为Mat的面积
         nwindows = cache.windowsInImage(paddedImgSize, winStride).area();
 
     const myHOGCache::BlockData* blockData = &cache.blockData[0];
 
     int nblocks = cache.nblocks.area();
     int blockHistogramSize = cache.blockHistogramSize;
     size_t dsize = getDescriptorSize();//一个hog的描述长度
     //resize()为改变矩阵的行数，如果减少矩阵的行数则只保留减少后的
     //那些行，如果是增加行数，则保留所有的行。
     //这里将描述子长度扩展到整幅图片
     descriptors.resize(dsize*nwindows);
 
     for( size_t i = 0; i < nwindows; i++ )
     {
         //descriptor为第i个检测窗口的描述子首位置。
         float* descriptor = &descriptors[i*dsize];
        
         Point pt0;
         //非空
         if( !locations.empty() )
         {
             pt0 = locations[i];
             //非法的点
             if( pt0.x < -padding.width || pt0.x > img.cols + padding.width - winSize.width ||
                 pt0.y < -padding.height || pt0.y > img.rows + padding.height - winSize.height )
                 continue;
         }
         //locations为空
         else
         {
             //pt0为没有扩充前图像对应的第i个检测窗口
             pt0 = cache.getWindow(paddedImgSize, winStride, (int)i).tl() - Point(padding);
             CV_Assert(pt0.x % cacheStride.width == 0 && pt0.y % cacheStride.height == 0);
         }
 
         for( int j = 0; j < nblocks; j++ )
         {
             const myHOGCache::BlockData& bj = blockData[j];
             //pt为block的左上角相对检测图片的坐标
             Point pt = pt0 + bj.imgOffset;
 
             //dst为该block在整个测试图片的描述子的位置
             float* dst = descriptor + bj.histOfs;
             const float* src = cache.getBlock(pt, dst);
             if( src != dst )
 #ifdef HAVE_IPP
                ippsCopy_32f(src,dst,blockHistogramSize);
 #else
                 for( int k = 0; k < blockHistogramSize; k++ )
                     dst[k] = src[k];
 #endif
         }
     }
 }

 void myHOGDescriptor::detect(const Mat& img,
     vector<Point>& hits, vector<double>& weights, double hitThreshold, 
     Size winStride, Size padding, const vector<Point>& locations) const
 {
     //hits里面存的是符合检测到目标的窗口的左上角顶点坐标
     hits.clear();
     if( svmDetector.empty() )
         return;
 
     if( winStride == Size() )
         winStride = cellSize;
     Size cacheStride(gcd(winStride.width, blockStride.width),
                      gcd(winStride.height, blockStride.height));
     size_t nwindows = locations.size();
     padding.width = (int)alignSize(std::max(padding.width, 0), cacheStride.width);
     padding.height = (int)alignSize(std::max(padding.height, 0), cacheStride.height);
     Size paddedImgSize(img.cols + padding.width*2, img.rows + padding.height*2);
 
     myHOGCache cache(this, img, padding, padding, nwindows == 0, cacheStride);
 
     if( !nwindows )
         nwindows = cache.windowsInImage(paddedImgSize, winStride).area();
 
     const myHOGCache::BlockData* blockData = &cache.blockData[0];
 
     int nblocks = cache.nblocks.area();
     int blockHistogramSize = cache.blockHistogramSize;
     size_t dsize = getDescriptorSize();
 
     double rho = svmDetector.size() > dsize ? svmDetector[dsize] : 0;
     vector<float> blockHist(blockHistogramSize);
 
     for( size_t i = 0; i < nwindows; i++ )
     {
         Point pt0;
         if( !locations.empty() )
         {
             pt0 = locations[i];
             if( pt0.x < -padding.width || pt0.x > img.cols + padding.width - winSize.width ||
                 pt0.y < -padding.height || pt0.y > img.rows + padding.height - winSize.height )
                 continue;
         }
         else
         {
             pt0 = cache.getWindow(paddedImgSize, winStride, (int)i).tl() - Point(padding);
             CV_Assert(pt0.x % cacheStride.width == 0 && pt0.y % cacheStride.height == 0);
         }
         double s = rho;
         //svmVec指向svmDetector最前面那个元素
         const float* svmVec = &svmDetector[0];
 #ifdef HAVE_IPP
         int j;
 #else
         int j, k;
 #endif
         for( j = 0; j < nblocks; j++, svmVec += blockHistogramSize )
         {
             const myHOGCache::BlockData& bj = blockData[j];
             Point pt = pt0 + bj.imgOffset;
             
             //vec为测试图片pt处的block贡献的描述子指针
             const float* vec = cache.getBlock(pt, &blockHist[0]);
 #ifdef HAVE_IPP
             Ipp32f partSum;
             ippsDotProd_32f(vec,svmVec,blockHistogramSize,&partSum);
             s += (double)partSum;
 #else
             for( k = 0; k <= blockHistogramSize - 4; k += 4 )
                 //const float* svmVec = &svmDetector[0];
                 s += vec[k]*svmVec[k] + vec[k+1]*svmVec[k+1] +
                     vec[k+2]*svmVec[k+2] + vec[k+3]*svmVec[k+3];
             for( ; k < blockHistogramSize; k++ )
                 s += vec[k]*svmVec[k];
 #endif
         }
         if( s >= hitThreshold )
         {
             hits.push_back(pt0);
             weights.push_back(s);
         }
     }
 }

void myHOGDescriptor::detect(const Mat& img, vector<Point>& hits, double hitThreshold, Size winStride, Size padding, const vector<Point>& locations) const
{
    vector<double> weightsV;
    detect(img, hits, weightsV, hitThreshold, winStride, padding, locations); // locations is empty here.
}

struct myHOGInvoker
{
    myHOGInvoker( const myHOGDescriptor* _hog, const Mat& _img,
                double _hitThreshold, Size _winStride, Size _padding,
                const double* _levelScale, ConcurrentRectVector* _vec, 
                ConcurrentDoubleVector* _weights=0, ConcurrentDoubleVector* _scales=0 ) 
    {
        hog = _hog;
        img = _img;
        hitThreshold = _hitThreshold;
        winStride = _winStride;
        padding = _padding;
        levelScale = _levelScale;
        vec = _vec;
        weights = _weights;
        scales = _scales;
    }

    void operator()( const BlockedRange& range ) const
    {
        int i, i1 = range.begin(), i2 = range.end();
        double minScale = i1 > 0 ? levelScale[i1] : i2 > 1 ? levelScale[i1+1] : std::max(img.cols, img.rows);
        Size maxSz(cvCeil(img.cols/minScale), cvCeil(img.rows/minScale));
        Mat smallerImgBuf(maxSz, img.type());
        vector<Point> locations;
        vector<double> hitsWeights;

        for( i = i1; i < i2; i++ )
        {
            double scale = levelScale[i];
            Size sz(cvRound(img.cols/scale), cvRound(img.rows/scale));
            Mat smallerImg(sz, img.type(), smallerImgBuf.data);
            if( sz == img.size() )
                smallerImg = Mat(sz, img.type(), img.data, img.step);
            else
                resize(img, smallerImg, sz);
            hog->detect(smallerImg, locations, hitsWeights, hitThreshold, 
				winStride, padding); // input is smallerImg
            Size scaledWinSize = Size(cvRound(hog->winSize.width*scale), cvRound(hog->winSize.height*scale));
            for( size_t j = 0; j < locations.size(); j++ )
            {
                vec->push_back(Rect(cvRound(locations[j].x*scale),
                                    cvRound(locations[j].y*scale),
                                    scaledWinSize.width, scaledWinSize.height));
                if (scales) {
                    scales->push_back(scale);
                }
            }
            
            if (weights && (!hitsWeights.empty()))
            {
                for (size_t j = 0; j < locations.size(); j++)
                {
                    weights->push_back(hitsWeights[j]);
                }
            }        
        }
    }

    const myHOGDescriptor* hog;
    Mat img;
    double hitThreshold;
    Size winStride;
    Size padding;
    const double* levelScale;
    ConcurrentRectVector* vec;
    ConcurrentDoubleVector* weights;
    ConcurrentDoubleVector* scales;
};


void myHOGDescriptor::detectMultiScale(
     const Mat& img, vector<Rect>& foundLocations, vector<double>& foundWeights,
     double hitThreshold, Size winStride, Size padding,
     double scale0, double finalThreshold, bool useMeanshiftGrouping) const  
 {
     double scale = 1.;
     int levels = 0;
 
     vector<double> levelScale;
 
     //nlevels默认的是64层
     for( levels = 0; levels < nlevels; levels++ )
     {
         levelScale.push_back(scale);
         if( cvRound(img.cols/scale) < winSize.width ||
             cvRound(img.rows/scale) < winSize.height ||
             scale0 <= 1 )
             break;
         //只考虑测试图片尺寸比检测窗口尺寸大的情况
         scale *= scale0;
     }
     levels = std::max(levels, 1);
     levelScale.resize(levels);
 
     ConcurrentRectVector allCandidates;
     ConcurrentDoubleVector tempScales;
     ConcurrentDoubleVector tempWeights;
     vector<double> foundScales;
     
     //TBB并行计算
     parallel_for(BlockedRange(0, (int)levelScale.size()),
                  myHOGInvoker(this, img, hitThreshold, winStride, padding, &levelScale[0], &allCandidates, &tempWeights, &tempScales));
     //将tempScales中的内容复制到foundScales中；back_inserter是指在指定参数迭代器的末尾插入数据
     std::copy(tempScales.begin(), tempScales.end(), back_inserter(foundScales));
     //容器的clear()方法是指移除容器中所有的数据
     foundLocations.clear();
     //将候选目标窗口保存在foundLocations中
     std::copy(allCandidates.begin(), allCandidates.end(), back_inserter(foundLocations));
     foundWeights.clear();
     //将候选目标可信度保存在foundWeights中
     std::copy(tempWeights.begin(), tempWeights.end(), back_inserter(foundWeights));
 
     if ( useMeanshiftGrouping )
     {
         groupRectangles_meanshift(foundLocations, foundWeights, foundScales, finalThreshold, winSize);
     }
     else
     {
         //对矩形框进行聚类
         groupRectangles(foundLocations, (int)finalThreshold, 0.2);
     }
 }

void myHOGDescriptor::detectMultiScale(
	const Mat& img, vector<Rect>& foundLocations, 
	double hitThreshold, Size winStride, Size padding,
    double scale0, double finalThreshold, bool useMeanshiftGrouping) const  
{
    vector<double> foundWeights;
    detectMultiScale(
		img, foundLocations, foundWeights, hitThreshold, winStride, 
        padding, scale0, finalThreshold, useMeanshiftGrouping);
}

void myHOGDescriptor::placeDetector()
{
	fftSvmDetector.resize(36);
	int xnum = (winSize.width - blockSize.width)/blockStride.width + 1;
	int ynum = (winSize.height - blockSize.height)/blockStride.height + 1;
	for(unsigned i = 0; i < 36; i++){
	//	fftSvmDetector[i].resize(xnum);
	//	for(unsigned j = 0; j < fftSvmDetector[i].size(); j++){
	//		fftSvmDetector[i][j].resize(ynum);
	//	}
		fftSvmDetector[i].create(ynum, xnum, CV_64FC2);
	}
	
	int posInBlock;
	int blockX;
	int blockY;
	for(unsigned i = 0; i < 36 * xnum * ynum; i++){
		posInBlock = i % 36;// just flip flop the block positions
		blockX = xnum - 1 - i / (36 * ynum);
		blockY = ynum - 1 - (i % (36 * ynum)) / 36;
		fftSvmDetector[posInBlock].at<std::complex<double> >(blockY, blockX) 
									= svmDetector[i];
	}

	for(unsigned k = 0; k < fftSvmDetector[0].cols; k++){
		for(unsigned j = 0; j < fftSvmDetector[0].rows; j++){
			for(unsigned i = 0; i < 36; i++){
				if(imag(fftSvmDetector[i].at<std::complex<double> >(j,k)))
					cout << "!!!" << endl;
			}
		}
	}
}

}
