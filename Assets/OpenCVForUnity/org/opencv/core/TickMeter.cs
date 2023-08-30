

using OpenCVForUnity.CoreModule;
using OpenCVForUnity.UtilsModule;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace OpenCVForUnity.CoreModule
{
    // C++: class TickMeter
    /**
     * a Class to measure passing time.
     *
     * The class computes passing time by counting the number of ticks per second. That is, the following code computes the
     * execution time in seconds:
     * <code>
     * TickMeter tm;
     * tm.start();
     * // do something ...
     * tm.stop();
     * std::cout &lt;&lt; tm.getTimeSec();
     * </code>
     *
     * It is also possible to compute the average time over multiple runs:
     * <code>
     * TickMeter tm;
     * for (int i = 0; i &lt; 100; i++)
     * {
     *     tm.start();
     *     // do something ...
     *     tm.stop();
     * }
     * double average_time = tm.getTimeSec() / tm.getCounter();
     * std::cout &lt;&lt; "Average time in second per iteration is: " &lt;&lt; average_time &lt;&lt; std::endl;
     * </code>
     * SEE: getTickCount, getTickFrequency
     */

    public class TickMeter : DisposableOpenCVObject
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
                        core_TickMeter_delete(nativeObj);
                    nativeObj = IntPtr.Zero;
                }
            }
            finally
            {
                base.Dispose(disposing);
            }

        }

        protected internal TickMeter(IntPtr addr) : base(addr) { }


        public IntPtr getNativeObjAddr() { return nativeObj; }

        // internal usage only
        public static TickMeter __fromPtr__(IntPtr addr) { return new TickMeter(addr); }

        //
        // C++:   cv::TickMeter::TickMeter()
        //

        public TickMeter()
        {


            nativeObj = core_TickMeter_TickMeter_10();


        }


        //
        // C++:  double cv::TickMeter::getTimeMicro()
        //

        /**
         * returns passed time in microseconds.
         * return automatically generated
         */
        public double getTimeMicro()
        {
            ThrowIfDisposed();

            return core_TickMeter_getTimeMicro_10(nativeObj);


        }


        //
        // C++:  double cv::TickMeter::getTimeMilli()
        //

        /**
         * returns passed time in milliseconds.
         * return automatically generated
         */
        public double getTimeMilli()
        {
            ThrowIfDisposed();

            return core_TickMeter_getTimeMilli_10(nativeObj);


        }


        //
        // C++:  double cv::TickMeter::getTimeSec()
        //

        /**
         * returns passed time in seconds.
         * return automatically generated
         */
        public double getTimeSec()
        {
            ThrowIfDisposed();

            return core_TickMeter_getTimeSec_10(nativeObj);


        }


        //
        // C++:  int64 cv::TickMeter::getCounter()
        //

        /**
         * returns internal counter value.
         * return automatically generated
         */
        public long getCounter()
        {
            ThrowIfDisposed();

            return core_TickMeter_getCounter_10(nativeObj);


        }


        //
        // C++:  int64 cv::TickMeter::getTimeTicks()
        //

        /**
         * returns counted ticks.
         * return automatically generated
         */
        public long getTimeTicks()
        {
            ThrowIfDisposed();

            return core_TickMeter_getTimeTicks_10(nativeObj);


        }


        //
        // C++:  void cv::TickMeter::reset()
        //

        /**
         * resets internal values.
         */
        public void reset()
        {
            ThrowIfDisposed();

            core_TickMeter_reset_10(nativeObj);


        }


        //
        // C++:  void cv::TickMeter::start()
        //

        /**
         * starts counting ticks.
         */
        public void start()
        {
            ThrowIfDisposed();

            core_TickMeter_start_10(nativeObj);


        }


        //
        // C++:  void cv::TickMeter::stop()
        //

        /**
         * stops counting ticks.
         */
        public void stop()
        {
            ThrowIfDisposed();

            core_TickMeter_stop_10(nativeObj);


        }


#if (UNITY_IOS || UNITY_WEBGL) && !UNITY_EDITOR
        const string LIBNAME = "__Internal";
#else
        const string LIBNAME = "opencvforunity";
#endif



        // C++:   cv::TickMeter::TickMeter()
        [DllImport(LIBNAME)]
        private static extern IntPtr core_TickMeter_TickMeter_10();

        // C++:  double cv::TickMeter::getTimeMicro()
        [DllImport(LIBNAME)]
        private static extern double core_TickMeter_getTimeMicro_10(IntPtr nativeObj);

        // C++:  double cv::TickMeter::getTimeMilli()
        [DllImport(LIBNAME)]
        private static extern double core_TickMeter_getTimeMilli_10(IntPtr nativeObj);

        // C++:  double cv::TickMeter::getTimeSec()
        [DllImport(LIBNAME)]
        private static extern double core_TickMeter_getTimeSec_10(IntPtr nativeObj);

        // C++:  int64 cv::TickMeter::getCounter()
        [DllImport(LIBNAME)]
        private static extern long core_TickMeter_getCounter_10(IntPtr nativeObj);

        // C++:  int64 cv::TickMeter::getTimeTicks()
        [DllImport(LIBNAME)]
        private static extern long core_TickMeter_getTimeTicks_10(IntPtr nativeObj);

        // C++:  void cv::TickMeter::reset()
        [DllImport(LIBNAME)]
        private static extern void core_TickMeter_reset_10(IntPtr nativeObj);

        // C++:  void cv::TickMeter::start()
        [DllImport(LIBNAME)]
        private static extern void core_TickMeter_start_10(IntPtr nativeObj);

        // C++:  void cv::TickMeter::stop()
        [DllImport(LIBNAME)]
        private static extern void core_TickMeter_stop_10(IntPtr nativeObj);

        // native support for java finalize()
        [DllImport(LIBNAME)]
        private static extern void core_TickMeter_delete(IntPtr nativeObj);

    }
}
