

using OpenCVForUnity.CoreModule;
using OpenCVForUnity.UtilsModule;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace OpenCVForUnity.Features2dModule
{
    // C++: class Params


    public class Params : DisposableOpenCVObject
    {

        protected override void Dispose(bool disposing)
        {

            try
            {
                if (disposing)
                {
                }
                if (IsEnabledDispose)
                {
                    if (nativeObj != IntPtr.Zero)
                        features2d_Params_delete(nativeObj);
                    nativeObj = IntPtr.Zero;
                }
            }
            finally
            {
                base.Dispose(disposing);
            }

        }

        protected internal Params(IntPtr addr) : base(addr) { }


        public IntPtr getNativeObjAddr() { return nativeObj; }

        // internal usage only
        public static Params __fromPtr__(IntPtr addr) { return new Params(addr); }

        //
        // C++:   cv::SimpleBlobDetector::Params::Params()
        //

        public Params()
        {


            nativeObj = features2d_Params_Params_10();


        }


        //
        // C++: float Params::thresholdStep
        //

        public float get_thresholdStep()
        {
            ThrowIfDisposed();

            return features2d_Params_get_1thresholdStep_10(nativeObj);


        }


        //
        // C++: void Params::thresholdStep
        //

        public void set_thresholdStep(float thresholdStep)
        {
            ThrowIfDisposed();

            features2d_Params_set_1thresholdStep_10(nativeObj, thresholdStep);


        }


        //
        // C++: float Params::minThreshold
        //

        public float get_minThreshold()
        {
            ThrowIfDisposed();

            return features2d_Params_get_1minThreshold_10(nativeObj);


        }


        //
        // C++: void Params::minThreshold
        //

        public void set_minThreshold(float minThreshold)
        {
            ThrowIfDisposed();

            features2d_Params_set_1minThreshold_10(nativeObj, minThreshold);


        }


        //
        // C++: float Params::maxThreshold
        //

        public float get_maxThreshold()
        {
            ThrowIfDisposed();

            return features2d_Params_get_1maxThreshold_10(nativeObj);


        }


        //
        // C++: void Params::maxThreshold
        //

        public void set_maxThreshold(float maxThreshold)
        {
            ThrowIfDisposed();

            features2d_Params_set_1maxThreshold_10(nativeObj, maxThreshold);


        }


        //
        // C++: size_t Params::minRepeatability
        //

        public long get_minRepeatability()
        {
            ThrowIfDisposed();

            return features2d_Params_get_1minRepeatability_10(nativeObj);


        }


        //
        // C++: void Params::minRepeatability
        //

        public void set_minRepeatability(long minRepeatability)
        {
            ThrowIfDisposed();

            features2d_Params_set_1minRepeatability_10(nativeObj, minRepeatability);


        }


        //
        // C++: float Params::minDistBetweenBlobs
        //

        public float get_minDistBetweenBlobs()
        {
            ThrowIfDisposed();

            return features2d_Params_get_1minDistBetweenBlobs_10(nativeObj);


        }


        //
        // C++: void Params::minDistBetweenBlobs
        //

        public void set_minDistBetweenBlobs(float minDistBetweenBlobs)
        {
            ThrowIfDisposed();

            features2d_Params_set_1minDistBetweenBlobs_10(nativeObj, minDistBetweenBlobs);


        }


        //
        // C++: bool Params::filterByColor
        //

        public bool get_filterByColor()
        {
            ThrowIfDisposed();

            return features2d_Params_get_1filterByColor_10(nativeObj);


        }


        //
        // C++: void Params::filterByColor
        //

        public void set_filterByColor(bool filterByColor)
        {
            ThrowIfDisposed();

            features2d_Params_set_1filterByColor_10(nativeObj, filterByColor);


        }


        //
        // C++: uchar Params::blobColor
        //

        // Return type 'uchar' is not supported, skipping the function


        //
        // C++: void Params::blobColor
        //

        // Unknown type 'uchar' (I), skipping the function


        //
        // C++: bool Params::filterByArea
        //

        public bool get_filterByArea()
        {
            ThrowIfDisposed();

            return features2d_Params_get_1filterByArea_10(nativeObj);


        }


        //
        // C++: void Params::filterByArea
        //

        public void set_filterByArea(bool filterByArea)
        {
            ThrowIfDisposed();

            features2d_Params_set_1filterByArea_10(nativeObj, filterByArea);


        }


        //
        // C++: float Params::minArea
        //

        public float get_minArea()
        {
            ThrowIfDisposed();

            return features2d_Params_get_1minArea_10(nativeObj);


        }


        //
        // C++: void Params::minArea
        //

        public void set_minArea(float minArea)
        {
            ThrowIfDisposed();

            features2d_Params_set_1minArea_10(nativeObj, minArea);


        }


        //
        // C++: float Params::maxArea
        //

        public float get_maxArea()
        {
            ThrowIfDisposed();

            return features2d_Params_get_1maxArea_10(nativeObj);


        }


        //
        // C++: void Params::maxArea
        //

        public void set_maxArea(float maxArea)
        {
            ThrowIfDisposed();

            features2d_Params_set_1maxArea_10(nativeObj, maxArea);


        }


        //
        // C++: bool Params::filterByCircularity
        //

        public bool get_filterByCircularity()
        {
            ThrowIfDisposed();

            return features2d_Params_get_1filterByCircularity_10(nativeObj);


        }


        //
        // C++: void Params::filterByCircularity
        //

        public void set_filterByCircularity(bool filterByCircularity)
        {
            ThrowIfDisposed();

            features2d_Params_set_1filterByCircularity_10(nativeObj, filterByCircularity);


        }


        //
        // C++: float Params::minCircularity
        //

        public float get_minCircularity()
        {
            ThrowIfDisposed();

            return features2d_Params_get_1minCircularity_10(nativeObj);


        }


        //
        // C++: void Params::minCircularity
        //

        public void set_minCircularity(float minCircularity)
        {
            ThrowIfDisposed();

            features2d_Params_set_1minCircularity_10(nativeObj, minCircularity);


        }


        //
        // C++: float Params::maxCircularity
        //

        public float get_maxCircularity()
        {
            ThrowIfDisposed();

            return features2d_Params_get_1maxCircularity_10(nativeObj);


        }


        //
        // C++: void Params::maxCircularity
        //

        public void set_maxCircularity(float maxCircularity)
        {
            ThrowIfDisposed();

            features2d_Params_set_1maxCircularity_10(nativeObj, maxCircularity);


        }


        //
        // C++: bool Params::filterByInertia
        //

        public bool get_filterByInertia()
        {
            ThrowIfDisposed();

            return features2d_Params_get_1filterByInertia_10(nativeObj);


        }


        //
        // C++: void Params::filterByInertia
        //

        public void set_filterByInertia(bool filterByInertia)
        {
            ThrowIfDisposed();

            features2d_Params_set_1filterByInertia_10(nativeObj, filterByInertia);


        }


        //
        // C++: float Params::minInertiaRatio
        //

        public float get_minInertiaRatio()
        {
            ThrowIfDisposed();

            return features2d_Params_get_1minInertiaRatio_10(nativeObj);


        }


        //
        // C++: void Params::minInertiaRatio
        //

        public void set_minInertiaRatio(float minInertiaRatio)
        {
            ThrowIfDisposed();

            features2d_Params_set_1minInertiaRatio_10(nativeObj, minInertiaRatio);


        }


        //
        // C++: float Params::maxInertiaRatio
        //

        public float get_maxInertiaRatio()
        {
            ThrowIfDisposed();

            return features2d_Params_get_1maxInertiaRatio_10(nativeObj);


        }


        //
        // C++: void Params::maxInertiaRatio
        //

        public void set_maxInertiaRatio(float maxInertiaRatio)
        {
            ThrowIfDisposed();

            features2d_Params_set_1maxInertiaRatio_10(nativeObj, maxInertiaRatio);


        }


        //
        // C++: bool Params::filterByConvexity
        //

        public bool get_filterByConvexity()
        {
            ThrowIfDisposed();

            return features2d_Params_get_1filterByConvexity_10(nativeObj);


        }


        //
        // C++: void Params::filterByConvexity
        //

        public void set_filterByConvexity(bool filterByConvexity)
        {
            ThrowIfDisposed();

            features2d_Params_set_1filterByConvexity_10(nativeObj, filterByConvexity);


        }


        //
        // C++: float Params::minConvexity
        //

        public float get_minConvexity()
        {
            ThrowIfDisposed();

            return features2d_Params_get_1minConvexity_10(nativeObj);


        }


        //
        // C++: void Params::minConvexity
        //

        public void set_minConvexity(float minConvexity)
        {
            ThrowIfDisposed();

            features2d_Params_set_1minConvexity_10(nativeObj, minConvexity);


        }


        //
        // C++: float Params::maxConvexity
        //

        public float get_maxConvexity()
        {
            ThrowIfDisposed();

            return features2d_Params_get_1maxConvexity_10(nativeObj);


        }


        //
        // C++: void Params::maxConvexity
        //

        public void set_maxConvexity(float maxConvexity)
        {
            ThrowIfDisposed();

            features2d_Params_set_1maxConvexity_10(nativeObj, maxConvexity);


        }


#if (UNITY_IOS || UNITY_WEBGL) && !UNITY_EDITOR
        const string LIBNAME = "__Internal";
#else
        const string LIBNAME = "opencvforunity";
#endif



        // C++:   cv::SimpleBlobDetector::Params::Params()
        [DllImport(LIBNAME)]
        private static extern IntPtr features2d_Params_Params_10();

        // C++: float Params::thresholdStep
        [DllImport(LIBNAME)]
        private static extern float features2d_Params_get_1thresholdStep_10(IntPtr nativeObj);

        // C++: void Params::thresholdStep
        [DllImport(LIBNAME)]
        private static extern void features2d_Params_set_1thresholdStep_10(IntPtr nativeObj, float thresholdStep);

        // C++: float Params::minThreshold
        [DllImport(LIBNAME)]
        private static extern float features2d_Params_get_1minThreshold_10(IntPtr nativeObj);

        // C++: void Params::minThreshold
        [DllImport(LIBNAME)]
        private static extern void features2d_Params_set_1minThreshold_10(IntPtr nativeObj, float minThreshold);

        // C++: float Params::maxThreshold
        [DllImport(LIBNAME)]
        private static extern float features2d_Params_get_1maxThreshold_10(IntPtr nativeObj);

        // C++: void Params::maxThreshold
        [DllImport(LIBNAME)]
        private static extern void features2d_Params_set_1maxThreshold_10(IntPtr nativeObj, float maxThreshold);

        // C++: size_t Params::minRepeatability
        [DllImport(LIBNAME)]
        private static extern long features2d_Params_get_1minRepeatability_10(IntPtr nativeObj);

        // C++: void Params::minRepeatability
        [DllImport(LIBNAME)]
        private static extern void features2d_Params_set_1minRepeatability_10(IntPtr nativeObj, long minRepeatability);

        // C++: float Params::minDistBetweenBlobs
        [DllImport(LIBNAME)]
        private static extern float features2d_Params_get_1minDistBetweenBlobs_10(IntPtr nativeObj);

        // C++: void Params::minDistBetweenBlobs
        [DllImport(LIBNAME)]
        private static extern void features2d_Params_set_1minDistBetweenBlobs_10(IntPtr nativeObj, float minDistBetweenBlobs);

        // C++: bool Params::filterByColor
        [DllImport(LIBNAME)]
        [return: MarshalAs(UnmanagedType.U1)]
        private static extern bool features2d_Params_get_1filterByColor_10(IntPtr nativeObj);

        // C++: void Params::filterByColor
        [DllImport(LIBNAME)]
        private static extern void features2d_Params_set_1filterByColor_10(IntPtr nativeObj, [MarshalAs(UnmanagedType.U1)] bool filterByColor);

        // C++: bool Params::filterByArea
        [DllImport(LIBNAME)]
        [return: MarshalAs(UnmanagedType.U1)]
        private static extern bool features2d_Params_get_1filterByArea_10(IntPtr nativeObj);

        // C++: void Params::filterByArea
        [DllImport(LIBNAME)]
        private static extern void features2d_Params_set_1filterByArea_10(IntPtr nativeObj, [MarshalAs(UnmanagedType.U1)] bool filterByArea);

        // C++: float Params::minArea
        [DllImport(LIBNAME)]
        private static extern float features2d_Params_get_1minArea_10(IntPtr nativeObj);

        // C++: void Params::minArea
        [DllImport(LIBNAME)]
        private static extern void features2d_Params_set_1minArea_10(IntPtr nativeObj, float minArea);

        // C++: float Params::maxArea
        [DllImport(LIBNAME)]
        private static extern float features2d_Params_get_1maxArea_10(IntPtr nativeObj);

        // C++: void Params::maxArea
        [DllImport(LIBNAME)]
        private static extern void features2d_Params_set_1maxArea_10(IntPtr nativeObj, float maxArea);

        // C++: bool Params::filterByCircularity
        [DllImport(LIBNAME)]
        [return: MarshalAs(UnmanagedType.U1)]
        private static extern bool features2d_Params_get_1filterByCircularity_10(IntPtr nativeObj);

        // C++: void Params::filterByCircularity
        [DllImport(LIBNAME)]
        private static extern void features2d_Params_set_1filterByCircularity_10(IntPtr nativeObj, [MarshalAs(UnmanagedType.U1)] bool filterByCircularity);

        // C++: float Params::minCircularity
        [DllImport(LIBNAME)]
        private static extern float features2d_Params_get_1minCircularity_10(IntPtr nativeObj);

        // C++: void Params::minCircularity
        [DllImport(LIBNAME)]
        private static extern void features2d_Params_set_1minCircularity_10(IntPtr nativeObj, float minCircularity);

        // C++: float Params::maxCircularity
        [DllImport(LIBNAME)]
        private static extern float features2d_Params_get_1maxCircularity_10(IntPtr nativeObj);

        // C++: void Params::maxCircularity
        [DllImport(LIBNAME)]
        private static extern void features2d_Params_set_1maxCircularity_10(IntPtr nativeObj, float maxCircularity);

        // C++: bool Params::filterByInertia
        [DllImport(LIBNAME)]
        [return: MarshalAs(UnmanagedType.U1)]
        private static extern bool features2d_Params_get_1filterByInertia_10(IntPtr nativeObj);

        // C++: void Params::filterByInertia
        [DllImport(LIBNAME)]
        private static extern void features2d_Params_set_1filterByInertia_10(IntPtr nativeObj, [MarshalAs(UnmanagedType.U1)] bool filterByInertia);

        // C++: float Params::minInertiaRatio
        [DllImport(LIBNAME)]
        private static extern float features2d_Params_get_1minInertiaRatio_10(IntPtr nativeObj);

        // C++: void Params::minInertiaRatio
        [DllImport(LIBNAME)]
        private static extern void features2d_Params_set_1minInertiaRatio_10(IntPtr nativeObj, float minInertiaRatio);

        // C++: float Params::maxInertiaRatio
        [DllImport(LIBNAME)]
        private static extern float features2d_Params_get_1maxInertiaRatio_10(IntPtr nativeObj);

        // C++: void Params::maxInertiaRatio
        [DllImport(LIBNAME)]
        private static extern void features2d_Params_set_1maxInertiaRatio_10(IntPtr nativeObj, float maxInertiaRatio);

        // C++: bool Params::filterByConvexity
        [DllImport(LIBNAME)]
        [return: MarshalAs(UnmanagedType.U1)]
        private static extern bool features2d_Params_get_1filterByConvexity_10(IntPtr nativeObj);

        // C++: void Params::filterByConvexity
        [DllImport(LIBNAME)]
        private static extern void features2d_Params_set_1filterByConvexity_10(IntPtr nativeObj, [MarshalAs(UnmanagedType.U1)] bool filterByConvexity);

        // C++: float Params::minConvexity
        [DllImport(LIBNAME)]
        private static extern float features2d_Params_get_1minConvexity_10(IntPtr nativeObj);

        // C++: void Params::minConvexity
        [DllImport(LIBNAME)]
        private static extern void features2d_Params_set_1minConvexity_10(IntPtr nativeObj, float minConvexity);

        // C++: float Params::maxConvexity
        [DllImport(LIBNAME)]
        private static extern float features2d_Params_get_1maxConvexity_10(IntPtr nativeObj);

        // C++: void Params::maxConvexity
        [DllImport(LIBNAME)]
        private static extern void features2d_Params_set_1maxConvexity_10(IntPtr nativeObj, float maxConvexity);

        // native support for java finalize()
        [DllImport(LIBNAME)]
        private static extern void features2d_Params_delete(IntPtr nativeObj);

    }
}
