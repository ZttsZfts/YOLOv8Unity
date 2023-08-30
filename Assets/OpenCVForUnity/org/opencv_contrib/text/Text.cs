#if !UNITY_WSA_10_0

using OpenCVForUnity.CoreModule;
using OpenCVForUnity.UtilsModule;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace OpenCVForUnity.TextModule
{
    // C++: class Text


    public class Text
    {

        // C++: enum <unnamed>
        public const int ERFILTER_NM_RGBLGrad = 0;
        public const int ERFILTER_NM_IHSGrad = 1;
        public const int OCR_LEVEL_WORD = 0;
        public const int OCR_LEVEL_TEXTLINE = 1;
        // C++: enum page_seg_mode
        public const int PSM_OSD_ONLY = 0;
        public const int PSM_AUTO_OSD = 1;
        public const int PSM_AUTO_ONLY = 2;
        public const int PSM_AUTO = 3;
        public const int PSM_SINGLE_COLUMN = 4;
        public const int PSM_SINGLE_BLOCK_VERT_TEXT = 5;
        public const int PSM_SINGLE_BLOCK = 6;
        public const int PSM_SINGLE_LINE = 7;
        public const int PSM_SINGLE_WORD = 8;
        public const int PSM_CIRCLE_WORD = 9;
        public const int PSM_SINGLE_CHAR = 10;
        // C++: enum classifier_type
        public const int OCR_KNN_CLASSIFIER = 0;
        public const int OCR_CNN_CLASSIFIER = 1;
        // C++: enum erGrouping_Modes
        public const int ERGROUPING_ORIENTATION_HORIZ = 0;
        public const int ERGROUPING_ORIENTATION_ANY = 1;
        // C++: enum ocr_engine_mode
        public const int OEM_TESSERACT_ONLY = 0;
        public const int OEM_CUBE_ONLY = 1;
        public const int OEM_TESSERACT_CUBE_COMBINED = 2;
        public const int OEM_DEFAULT = 3;
        // C++: enum decoder_mode
        public const int OCR_DECODER_VITERBI = 0;
        //
        // C++:  Mat cv::text::createOCRHMMTransitionsTable(String vocabulary, vector_String lexicon)
        //

        /**
         * Utility function to create a tailored language model transitions table from a given list of words (lexicon).
         *
         * param vocabulary The language vocabulary (chars when ASCII English text).
         *
         * param lexicon The list of words that are expected to be found in a particular image.
         *
         *
         * The function calculate frequency statistics of character pairs from the given lexicon and fills the output transition_probabilities_table with them. The transition_probabilities_table can be used as input in the OCRHMMDecoder::create() and OCRBeamSearchDecoder::create() methods.
         * <b>Note:</b>
         * -   (C++) An alternative would be to load the default generic language transition table provided in the text module samples folder (created from ispell 42869 english words list) :
         * &lt;https://github.com/opencv/opencv_contrib/blob/master/modules/text/samples/OCRHMM_transitions_table.xml&gt;
         *
         * return automatically generated
         */
        public static Mat createOCRHMMTransitionsTable(string vocabulary, List<string> lexicon)
        {

            Mat lexicon_mat = Converters.vector_String_to_Mat(lexicon);
            return new Mat(text_Text_createOCRHMMTransitionsTable_10(vocabulary, lexicon_mat.nativeObj));


        }


        //
        // C++:  Ptr_ERFilter cv::text::createERFilterNM1(Ptr_ERFilter_Callback cb, int thresholdDelta = 1, float minArea = (float)0.00025, float maxArea = (float)0.13, float minProbability = (float)0.4, bool nonMaxSuppression = true, float minProbabilityDiff = (float)0.1)
        //

        // Unknown type 'Ptr_ERFilter_Callback' (I), skipping the function


        //
        // C++:  Ptr_ERFilter cv::text::createERFilterNM1(String filename, int thresholdDelta = 1, float minArea = (float)0.00025, float maxArea = (float)0.13, float minProbability = (float)0.4, bool nonMaxSuppression = true, float minProbabilityDiff = (float)0.1)
        //

        /**
         * Reads an Extremal Region Filter for the 1st stage classifier of N&amp;M algorithm
         *     from the provided path e.g. /path/to/cpp/trained_classifierNM1.xml
         *
         * param filename automatically generated
         * param thresholdDelta automatically generated
         * param minArea automatically generated
         * param maxArea automatically generated
         * param minProbability automatically generated
         * param nonMaxSuppression automatically generated
         * param minProbabilityDiff automatically generated
         * return automatically generated
         */
        public static ERFilter createERFilterNM1(string filename, int thresholdDelta, float minArea, float maxArea, float minProbability, bool nonMaxSuppression, float minProbabilityDiff)
        {


            return ERFilter.__fromPtr__(text_Text_createERFilterNM1_10(filename, thresholdDelta, minArea, maxArea, minProbability, nonMaxSuppression, minProbabilityDiff));


        }

        /**
         * Reads an Extremal Region Filter for the 1st stage classifier of N&amp;M algorithm
         *     from the provided path e.g. /path/to/cpp/trained_classifierNM1.xml
         *
         * param filename automatically generated
         * param thresholdDelta automatically generated
         * param minArea automatically generated
         * param maxArea automatically generated
         * param minProbability automatically generated
         * param nonMaxSuppression automatically generated
         * return automatically generated
         */
        public static ERFilter createERFilterNM1(string filename, int thresholdDelta, float minArea, float maxArea, float minProbability, bool nonMaxSuppression)
        {


            return ERFilter.__fromPtr__(text_Text_createERFilterNM1_11(filename, thresholdDelta, minArea, maxArea, minProbability, nonMaxSuppression));


        }

        /**
         * Reads an Extremal Region Filter for the 1st stage classifier of N&amp;M algorithm
         *     from the provided path e.g. /path/to/cpp/trained_classifierNM1.xml
         *
         * param filename automatically generated
         * param thresholdDelta automatically generated
         * param minArea automatically generated
         * param maxArea automatically generated
         * param minProbability automatically generated
         * return automatically generated
         */
        public static ERFilter createERFilterNM1(string filename, int thresholdDelta, float minArea, float maxArea, float minProbability)
        {


            return ERFilter.__fromPtr__(text_Text_createERFilterNM1_12(filename, thresholdDelta, minArea, maxArea, minProbability));


        }

        /**
         * Reads an Extremal Region Filter for the 1st stage classifier of N&amp;M algorithm
         *     from the provided path e.g. /path/to/cpp/trained_classifierNM1.xml
         *
         * param filename automatically generated
         * param thresholdDelta automatically generated
         * param minArea automatically generated
         * param maxArea automatically generated
         * return automatically generated
         */
        public static ERFilter createERFilterNM1(string filename, int thresholdDelta, float minArea, float maxArea)
        {


            return ERFilter.__fromPtr__(text_Text_createERFilterNM1_13(filename, thresholdDelta, minArea, maxArea));


        }

        /**
         * Reads an Extremal Region Filter for the 1st stage classifier of N&amp;M algorithm
         *     from the provided path e.g. /path/to/cpp/trained_classifierNM1.xml
         *
         * param filename automatically generated
         * param thresholdDelta automatically generated
         * param minArea automatically generated
         * return automatically generated
         */
        public static ERFilter createERFilterNM1(string filename, int thresholdDelta, float minArea)
        {


            return ERFilter.__fromPtr__(text_Text_createERFilterNM1_14(filename, thresholdDelta, minArea));


        }

        /**
         * Reads an Extremal Region Filter for the 1st stage classifier of N&amp;M algorithm
         *     from the provided path e.g. /path/to/cpp/trained_classifierNM1.xml
         *
         * param filename automatically generated
         * param thresholdDelta automatically generated
         * return automatically generated
         */
        public static ERFilter createERFilterNM1(string filename, int thresholdDelta)
        {


            return ERFilter.__fromPtr__(text_Text_createERFilterNM1_15(filename, thresholdDelta));


        }

        /**
         * Reads an Extremal Region Filter for the 1st stage classifier of N&amp;M algorithm
         *     from the provided path e.g. /path/to/cpp/trained_classifierNM1.xml
         *
         * param filename automatically generated
         * return automatically generated
         */
        public static ERFilter createERFilterNM1(string filename)
        {


            return ERFilter.__fromPtr__(text_Text_createERFilterNM1_16(filename));


        }


        //
        // C++:  Ptr_ERFilter cv::text::createERFilterNM2(Ptr_ERFilter_Callback cb, float minProbability = (float)0.3)
        //

        // Unknown type 'Ptr_ERFilter_Callback' (I), skipping the function


        //
        // C++:  Ptr_ERFilter cv::text::createERFilterNM2(String filename, float minProbability = (float)0.3)
        //

        /**
         * Reads an Extremal Region Filter for the 2nd stage classifier of N&amp;M algorithm
         *     from the provided path e.g. /path/to/cpp/trained_classifierNM2.xml
         *
         * param filename automatically generated
         * param minProbability automatically generated
         * return automatically generated
         */
        public static ERFilter createERFilterNM2(string filename, float minProbability)
        {


            return ERFilter.__fromPtr__(text_Text_createERFilterNM2_10(filename, minProbability));


        }

        /**
         * Reads an Extremal Region Filter for the 2nd stage classifier of N&amp;M algorithm
         *     from the provided path e.g. /path/to/cpp/trained_classifierNM2.xml
         *
         * param filename automatically generated
         * return automatically generated
         */
        public static ERFilter createERFilterNM2(string filename)
        {


            return ERFilter.__fromPtr__(text_Text_createERFilterNM2_11(filename));


        }


        //
        // C++:  Ptr_ERFilter_Callback cv::text::loadClassifierNM1(String filename)
        //

        // Return type 'Ptr_ERFilter_Callback' is not supported, skipping the function


        //
        // C++:  Ptr_ERFilter_Callback cv::text::loadClassifierNM2(String filename)
        //

        // Return type 'Ptr_ERFilter_Callback' is not supported, skipping the function


        //
        // C++:  Ptr_OCRBeamSearchDecoder_ClassifierCallback cv::text::loadOCRBeamSearchClassifierCNN(String filename)
        //

        // Return type 'Ptr_OCRBeamSearchDecoder_ClassifierCallback' is not supported, skipping the function


        //
        // C++:  Ptr_OCRHMMDecoder_ClassifierCallback cv::text::loadOCRHMMClassifier(String filename, int classifier)
        //

        // Return type 'Ptr_OCRHMMDecoder_ClassifierCallback' is not supported, skipping the function


        //
        // C++:  Ptr_OCRHMMDecoder_ClassifierCallback cv::text::loadOCRHMMClassifierCNN(String filename)
        //

        // Return type 'Ptr_OCRHMMDecoder_ClassifierCallback' is not supported, skipping the function


        //
        // C++:  Ptr_OCRHMMDecoder_ClassifierCallback cv::text::loadOCRHMMClassifierNM(String filename)
        //

        // Return type 'Ptr_OCRHMMDecoder_ClassifierCallback' is not supported, skipping the function


        //
        // C++:  void cv::text::computeNMChannels(Mat _src, vector_Mat& _channels, int _mode = ERFILTER_NM_RGBLGrad)
        //

        /**
         * Compute the different channels to be processed independently in the N&amp;M algorithm CITE: Neumann12.
         *
         * param _src Source image. Must be RGB CV_8UC3.
         *
         * param _channels Output vector&lt;Mat&gt; where computed channels are stored.
         *
         * param _mode Mode of operation. Currently the only available options are:
         * <b>ERFILTER_NM_RGBLGrad</b> (used by default) and <b>ERFILTER_NM_IHSGrad</b>.
         *
         * In N&amp;M algorithm, the combination of intensity (I), hue (H), saturation (S), and gradient magnitude
         * channels (Grad) are used in order to obtain high localization recall. This implementation also
         * provides an alternative combination of red (R), green (G), blue (B), lightness (L), and gradient
         * magnitude (Grad).
         */
        public static void computeNMChannels(Mat _src, List<Mat> _channels, int _mode)
        {
            if (_src != null) _src.ThrowIfDisposed();
            Mat _channels_mat = new Mat();
            text_Text_computeNMChannels_10(_src.nativeObj, _channels_mat.nativeObj, _mode);
            Converters.Mat_to_vector_Mat(_channels_mat, _channels);
            _channels_mat.release();

        }

        /**
         * Compute the different channels to be processed independently in the N&amp;M algorithm CITE: Neumann12.
         *
         * param _src Source image. Must be RGB CV_8UC3.
         *
         * param _channels Output vector&lt;Mat&gt; where computed channels are stored.
         *
         * <b>ERFILTER_NM_RGBLGrad</b> (used by default) and <b>ERFILTER_NM_IHSGrad</b>.
         *
         * In N&amp;M algorithm, the combination of intensity (I), hue (H), saturation (S), and gradient magnitude
         * channels (Grad) are used in order to obtain high localization recall. This implementation also
         * provides an alternative combination of red (R), green (G), blue (B), lightness (L), and gradient
         * magnitude (Grad).
         */
        public static void computeNMChannels(Mat _src, List<Mat> _channels)
        {
            if (_src != null) _src.ThrowIfDisposed();
            Mat _channels_mat = new Mat();
            text_Text_computeNMChannels_11(_src.nativeObj, _channels_mat.nativeObj);
            Converters.Mat_to_vector_Mat(_channels_mat, _channels);
            _channels_mat.release();

        }


        //
        // C++:  void cv::text::detectRegions(Mat image, Ptr_ERFilter er_filter1, Ptr_ERFilter er_filter2, vector_Rect& groups_rects, int method = ERGROUPING_ORIENTATION_HORIZ, String filename = String(), float minProbability = (float)0.5)
        //

        /**
         * Extracts text regions from image.
         *
         * param image Source image where text blocks needs to be extracted from.  Should be CV_8UC3 (color).
         * param er_filter1 Extremal Region Filter for the 1st stage classifier of N&amp;M algorithm CITE: Neumann12
         * param er_filter2 Extremal Region Filter for the 2nd stage classifier of N&amp;M algorithm CITE: Neumann12
         * param groups_rects Output list of rectangle blocks with text
         * param method Grouping method (see text::erGrouping_Modes). Can be one of ERGROUPING_ORIENTATION_HORIZ, ERGROUPING_ORIENTATION_ANY.
         * param filename The XML or YAML file with the classifier model (e.g. samples/trained_classifier_erGrouping.xml). Only to use when grouping method is ERGROUPING_ORIENTATION_ANY.
         * param minProbability The minimum probability for accepting a group. Only to use when grouping method is ERGROUPING_ORIENTATION_ANY.
         */
        public static void detectRegions(Mat image, ERFilter er_filter1, ERFilter er_filter2, MatOfRect groups_rects, int method, string filename, float minProbability)
        {
            if (image != null) image.ThrowIfDisposed();
            if (er_filter1 != null) er_filter1.ThrowIfDisposed();
            if (er_filter2 != null) er_filter2.ThrowIfDisposed();
            if (groups_rects != null) groups_rects.ThrowIfDisposed();
            Mat groups_rects_mat = groups_rects;
            text_Text_detectRegions_10(image.nativeObj, er_filter1.getNativeObjAddr(), er_filter2.getNativeObjAddr(), groups_rects_mat.nativeObj, method, filename, minProbability);


        }

        /**
         * Extracts text regions from image.
         *
         * param image Source image where text blocks needs to be extracted from.  Should be CV_8UC3 (color).
         * param er_filter1 Extremal Region Filter for the 1st stage classifier of N&amp;M algorithm CITE: Neumann12
         * param er_filter2 Extremal Region Filter for the 2nd stage classifier of N&amp;M algorithm CITE: Neumann12
         * param groups_rects Output list of rectangle blocks with text
         * param method Grouping method (see text::erGrouping_Modes). Can be one of ERGROUPING_ORIENTATION_HORIZ, ERGROUPING_ORIENTATION_ANY.
         * param filename The XML or YAML file with the classifier model (e.g. samples/trained_classifier_erGrouping.xml). Only to use when grouping method is ERGROUPING_ORIENTATION_ANY.
         */
        public static void detectRegions(Mat image, ERFilter er_filter1, ERFilter er_filter2, MatOfRect groups_rects, int method, string filename)
        {
            if (image != null) image.ThrowIfDisposed();
            if (er_filter1 != null) er_filter1.ThrowIfDisposed();
            if (er_filter2 != null) er_filter2.ThrowIfDisposed();
            if (groups_rects != null) groups_rects.ThrowIfDisposed();
            Mat groups_rects_mat = groups_rects;
            text_Text_detectRegions_11(image.nativeObj, er_filter1.getNativeObjAddr(), er_filter2.getNativeObjAddr(), groups_rects_mat.nativeObj, method, filename);


        }

        /**
         * Extracts text regions from image.
         *
         * param image Source image where text blocks needs to be extracted from.  Should be CV_8UC3 (color).
         * param er_filter1 Extremal Region Filter for the 1st stage classifier of N&amp;M algorithm CITE: Neumann12
         * param er_filter2 Extremal Region Filter for the 2nd stage classifier of N&amp;M algorithm CITE: Neumann12
         * param groups_rects Output list of rectangle blocks with text
         * param method Grouping method (see text::erGrouping_Modes). Can be one of ERGROUPING_ORIENTATION_HORIZ, ERGROUPING_ORIENTATION_ANY.
         */
        public static void detectRegions(Mat image, ERFilter er_filter1, ERFilter er_filter2, MatOfRect groups_rects, int method)
        {
            if (image != null) image.ThrowIfDisposed();
            if (er_filter1 != null) er_filter1.ThrowIfDisposed();
            if (er_filter2 != null) er_filter2.ThrowIfDisposed();
            if (groups_rects != null) groups_rects.ThrowIfDisposed();
            Mat groups_rects_mat = groups_rects;
            text_Text_detectRegions_12(image.nativeObj, er_filter1.getNativeObjAddr(), er_filter2.getNativeObjAddr(), groups_rects_mat.nativeObj, method);


        }

        /**
         * Extracts text regions from image.
         *
         * param image Source image where text blocks needs to be extracted from.  Should be CV_8UC3 (color).
         * param er_filter1 Extremal Region Filter for the 1st stage classifier of N&amp;M algorithm CITE: Neumann12
         * param er_filter2 Extremal Region Filter for the 2nd stage classifier of N&amp;M algorithm CITE: Neumann12
         * param groups_rects Output list of rectangle blocks with text
         */
        public static void detectRegions(Mat image, ERFilter er_filter1, ERFilter er_filter2, MatOfRect groups_rects)
        {
            if (image != null) image.ThrowIfDisposed();
            if (er_filter1 != null) er_filter1.ThrowIfDisposed();
            if (er_filter2 != null) er_filter2.ThrowIfDisposed();
            if (groups_rects != null) groups_rects.ThrowIfDisposed();
            Mat groups_rects_mat = groups_rects;
            text_Text_detectRegions_13(image.nativeObj, er_filter1.getNativeObjAddr(), er_filter2.getNativeObjAddr(), groups_rects_mat.nativeObj);


        }


        //
        // C++:  void cv::text::detectRegions(Mat image, Ptr_ERFilter er_filter1, Ptr_ERFilter er_filter2, vector_vector_Point& regions)
        //

        /**
         * Converts MSER contours (vector&lt;Point&gt;) to ERStat regions.
         *
         * param image Source image CV_8UC1 from which the MSERs where extracted.
         *
         *
         * param regions Output where the ERStat regions are stored.
         *
         * It takes as input the contours provided by the OpenCV MSER feature detector and returns as output
         * two vectors of ERStats. This is because MSER() output contains both MSER+ and MSER- regions in a
         * single vector&lt;Point&gt;, the function separates them in two different vectors (this is as if the
         * ERStats where extracted from two different channels).
         *
         * An example of MSERsToERStats in use can be found in the text detection webcam_demo:
         * &lt;https://github.com/opencv/opencv_contrib/blob/master/modules/text/samples/webcam_demo.cpp&gt;
         * param er_filter1 automatically generated
         * param er_filter2 automatically generated
         */
        public static void detectRegions(Mat image, ERFilter er_filter1, ERFilter er_filter2, List<MatOfPoint> regions)
        {
            if (image != null) image.ThrowIfDisposed();
            if (er_filter1 != null) er_filter1.ThrowIfDisposed();
            if (er_filter2 != null) er_filter2.ThrowIfDisposed();
            Mat regions_mat = new Mat();
            text_Text_detectRegions_14(image.nativeObj, er_filter1.getNativeObjAddr(), er_filter2.getNativeObjAddr(), regions_mat.nativeObj);
            Converters.Mat_to_vector_vector_Point(regions_mat, regions);
            regions_mat.release();

        }


        //
        // C++:  void cv::text::erGrouping(Mat image, Mat channel, vector_vector_Point regions, vector_Rect& groups_rects, int method = ERGROUPING_ORIENTATION_HORIZ, String filename = String(), float minProbablity = (float)0.5)
        //

        /**
         * Find groups of Extremal Regions that are organized as text blocks.
         *
         *
         *
         * param regions Vector of ER's retrieved from the ERFilter algorithm from each channel.
         *
         * provided regions.
         *
         * param groups_rects The output of the algorithm are stored in this parameter as list of rectangles.
         *
         * param method Grouping method (see text::erGrouping_Modes). Can be one of ERGROUPING_ORIENTATION_HORIZ,
         * ERGROUPING_ORIENTATION_ANY.
         *
         * param filename The XML or YAML file with the classifier model (e.g.
         * samples/trained_classifier_erGrouping.xml). Only to use when grouping method is
         * ERGROUPING_ORIENTATION_ANY.
         *
         * param minProbablity The minimum probability for accepting a group. Only to use when grouping
         * method is ERGROUPING_ORIENTATION_ANY.
         * param image automatically generated
         * param channel automatically generated
         */
        public static void erGrouping(Mat image, Mat channel, List<MatOfPoint> regions, MatOfRect groups_rects, int method, string filename, float minProbablity)
        {
            if (image != null) image.ThrowIfDisposed();
            if (channel != null) channel.ThrowIfDisposed();
            if (groups_rects != null) groups_rects.ThrowIfDisposed();
            List<Mat> regions_tmplm = new List<Mat>((regions != null) ? regions.Count : 0);
            Mat regions_mat = Converters.vector_vector_Point_to_Mat(regions, regions_tmplm);
            Mat groups_rects_mat = groups_rects;
            text_Text_erGrouping_10(image.nativeObj, channel.nativeObj, regions_mat.nativeObj, groups_rects_mat.nativeObj, method, filename, minProbablity);


        }

        /**
         * Find groups of Extremal Regions that are organized as text blocks.
         *
         *
         *
         * param regions Vector of ER's retrieved from the ERFilter algorithm from each channel.
         *
         * provided regions.
         *
         * param groups_rects The output of the algorithm are stored in this parameter as list of rectangles.
         *
         * param method Grouping method (see text::erGrouping_Modes). Can be one of ERGROUPING_ORIENTATION_HORIZ,
         * ERGROUPING_ORIENTATION_ANY.
         *
         * param filename The XML or YAML file with the classifier model (e.g.
         * samples/trained_classifier_erGrouping.xml). Only to use when grouping method is
         * ERGROUPING_ORIENTATION_ANY.
         *
         * method is ERGROUPING_ORIENTATION_ANY.
         * param image automatically generated
         * param channel automatically generated
         */
        public static void erGrouping(Mat image, Mat channel, List<MatOfPoint> regions, MatOfRect groups_rects, int method, string filename)
        {
            if (image != null) image.ThrowIfDisposed();
            if (channel != null) channel.ThrowIfDisposed();
            if (groups_rects != null) groups_rects.ThrowIfDisposed();
            List<Mat> regions_tmplm = new List<Mat>((regions != null) ? regions.Count : 0);
            Mat regions_mat = Converters.vector_vector_Point_to_Mat(regions, regions_tmplm);
            Mat groups_rects_mat = groups_rects;
            text_Text_erGrouping_11(image.nativeObj, channel.nativeObj, regions_mat.nativeObj, groups_rects_mat.nativeObj, method, filename);


        }

        /**
         * Find groups of Extremal Regions that are organized as text blocks.
         *
         *
         *
         * param regions Vector of ER's retrieved from the ERFilter algorithm from each channel.
         *
         * provided regions.
         *
         * param groups_rects The output of the algorithm are stored in this parameter as list of rectangles.
         *
         * param method Grouping method (see text::erGrouping_Modes). Can be one of ERGROUPING_ORIENTATION_HORIZ,
         * ERGROUPING_ORIENTATION_ANY.
         *
         * samples/trained_classifier_erGrouping.xml). Only to use when grouping method is
         * ERGROUPING_ORIENTATION_ANY.
         *
         * method is ERGROUPING_ORIENTATION_ANY.
         * param image automatically generated
         * param channel automatically generated
         */
        public static void erGrouping(Mat image, Mat channel, List<MatOfPoint> regions, MatOfRect groups_rects, int method)
        {
            if (image != null) image.ThrowIfDisposed();
            if (channel != null) channel.ThrowIfDisposed();
            if (groups_rects != null) groups_rects.ThrowIfDisposed();
            List<Mat> regions_tmplm = new List<Mat>((regions != null) ? regions.Count : 0);
            Mat regions_mat = Converters.vector_vector_Point_to_Mat(regions, regions_tmplm);
            Mat groups_rects_mat = groups_rects;
            text_Text_erGrouping_12(image.nativeObj, channel.nativeObj, regions_mat.nativeObj, groups_rects_mat.nativeObj, method);


        }

        /**
         * Find groups of Extremal Regions that are organized as text blocks.
         *
         *
         *
         * param regions Vector of ER's retrieved from the ERFilter algorithm from each channel.
         *
         * provided regions.
         *
         * param groups_rects The output of the algorithm are stored in this parameter as list of rectangles.
         *
         * ERGROUPING_ORIENTATION_ANY.
         *
         * samples/trained_classifier_erGrouping.xml). Only to use when grouping method is
         * ERGROUPING_ORIENTATION_ANY.
         *
         * method is ERGROUPING_ORIENTATION_ANY.
         * param image automatically generated
         * param channel automatically generated
         */
        public static void erGrouping(Mat image, Mat channel, List<MatOfPoint> regions, MatOfRect groups_rects)
        {
            if (image != null) image.ThrowIfDisposed();
            if (channel != null) channel.ThrowIfDisposed();
            if (groups_rects != null) groups_rects.ThrowIfDisposed();
            List<Mat> regions_tmplm = new List<Mat>((regions != null) ? regions.Count : 0);
            Mat regions_mat = Converters.vector_vector_Point_to_Mat(regions, regions_tmplm);
            Mat groups_rects_mat = groups_rects;
            text_Text_erGrouping_13(image.nativeObj, channel.nativeObj, regions_mat.nativeObj, groups_rects_mat.nativeObj);


        }


#if (UNITY_IOS || UNITY_WEBGL) && !UNITY_EDITOR
        const string LIBNAME = "__Internal";
#else
        const string LIBNAME = "opencvforunity";
#endif



        // C++:  Mat cv::text::createOCRHMMTransitionsTable(String vocabulary, vector_String lexicon)
        [DllImport(LIBNAME)]
        private static extern IntPtr text_Text_createOCRHMMTransitionsTable_10(string vocabulary, IntPtr lexicon_mat_nativeObj);

        // C++:  Ptr_ERFilter cv::text::createERFilterNM1(String filename, int thresholdDelta = 1, float minArea = (float)0.00025, float maxArea = (float)0.13, float minProbability = (float)0.4, bool nonMaxSuppression = true, float minProbabilityDiff = (float)0.1)
        [DllImport(LIBNAME)]
        private static extern IntPtr text_Text_createERFilterNM1_10(string filename, int thresholdDelta, float minArea, float maxArea, float minProbability, [MarshalAs(UnmanagedType.U1)] bool nonMaxSuppression, float minProbabilityDiff);
        [DllImport(LIBNAME)]
        private static extern IntPtr text_Text_createERFilterNM1_11(string filename, int thresholdDelta, float minArea, float maxArea, float minProbability, [MarshalAs(UnmanagedType.U1)] bool nonMaxSuppression);
        [DllImport(LIBNAME)]
        private static extern IntPtr text_Text_createERFilterNM1_12(string filename, int thresholdDelta, float minArea, float maxArea, float minProbability);
        [DllImport(LIBNAME)]
        private static extern IntPtr text_Text_createERFilterNM1_13(string filename, int thresholdDelta, float minArea, float maxArea);
        [DllImport(LIBNAME)]
        private static extern IntPtr text_Text_createERFilterNM1_14(string filename, int thresholdDelta, float minArea);
        [DllImport(LIBNAME)]
        private static extern IntPtr text_Text_createERFilterNM1_15(string filename, int thresholdDelta);
        [DllImport(LIBNAME)]
        private static extern IntPtr text_Text_createERFilterNM1_16(string filename);

        // C++:  Ptr_ERFilter cv::text::createERFilterNM2(String filename, float minProbability = (float)0.3)
        [DllImport(LIBNAME)]
        private static extern IntPtr text_Text_createERFilterNM2_10(string filename, float minProbability);
        [DllImport(LIBNAME)]
        private static extern IntPtr text_Text_createERFilterNM2_11(string filename);

        // C++:  void cv::text::computeNMChannels(Mat _src, vector_Mat& _channels, int _mode = ERFILTER_NM_RGBLGrad)
        [DllImport(LIBNAME)]
        private static extern void text_Text_computeNMChannels_10(IntPtr _src_nativeObj, IntPtr _channels_mat_nativeObj, int _mode);
        [DllImport(LIBNAME)]
        private static extern void text_Text_computeNMChannels_11(IntPtr _src_nativeObj, IntPtr _channels_mat_nativeObj);

        // C++:  void cv::text::detectRegions(Mat image, Ptr_ERFilter er_filter1, Ptr_ERFilter er_filter2, vector_Rect& groups_rects, int method = ERGROUPING_ORIENTATION_HORIZ, String filename = String(), float minProbability = (float)0.5)
        [DllImport(LIBNAME)]
        private static extern void text_Text_detectRegions_10(IntPtr image_nativeObj, IntPtr er_filter1_nativeObj, IntPtr er_filter2_nativeObj, IntPtr groups_rects_mat_nativeObj, int method, string filename, float minProbability);
        [DllImport(LIBNAME)]
        private static extern void text_Text_detectRegions_11(IntPtr image_nativeObj, IntPtr er_filter1_nativeObj, IntPtr er_filter2_nativeObj, IntPtr groups_rects_mat_nativeObj, int method, string filename);
        [DllImport(LIBNAME)]
        private static extern void text_Text_detectRegions_12(IntPtr image_nativeObj, IntPtr er_filter1_nativeObj, IntPtr er_filter2_nativeObj, IntPtr groups_rects_mat_nativeObj, int method);
        [DllImport(LIBNAME)]
        private static extern void text_Text_detectRegions_13(IntPtr image_nativeObj, IntPtr er_filter1_nativeObj, IntPtr er_filter2_nativeObj, IntPtr groups_rects_mat_nativeObj);

        // C++:  void cv::text::detectRegions(Mat image, Ptr_ERFilter er_filter1, Ptr_ERFilter er_filter2, vector_vector_Point& regions)
        [DllImport(LIBNAME)]
        private static extern void text_Text_detectRegions_14(IntPtr image_nativeObj, IntPtr er_filter1_nativeObj, IntPtr er_filter2_nativeObj, IntPtr regions_mat_nativeObj);

        // C++:  void cv::text::erGrouping(Mat image, Mat channel, vector_vector_Point regions, vector_Rect& groups_rects, int method = ERGROUPING_ORIENTATION_HORIZ, String filename = String(), float minProbablity = (float)0.5)
        [DllImport(LIBNAME)]
        private static extern void text_Text_erGrouping_10(IntPtr image_nativeObj, IntPtr channel_nativeObj, IntPtr regions_mat_nativeObj, IntPtr groups_rects_mat_nativeObj, int method, string filename, float minProbablity);
        [DllImport(LIBNAME)]
        private static extern void text_Text_erGrouping_11(IntPtr image_nativeObj, IntPtr channel_nativeObj, IntPtr regions_mat_nativeObj, IntPtr groups_rects_mat_nativeObj, int method, string filename);
        [DllImport(LIBNAME)]
        private static extern void text_Text_erGrouping_12(IntPtr image_nativeObj, IntPtr channel_nativeObj, IntPtr regions_mat_nativeObj, IntPtr groups_rects_mat_nativeObj, int method);
        [DllImport(LIBNAME)]
        private static extern void text_Text_erGrouping_13(IntPtr image_nativeObj, IntPtr channel_nativeObj, IntPtr regions_mat_nativeObj, IntPtr groups_rects_mat_nativeObj);

    }
}
#endif