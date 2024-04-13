package com.programminghut.pose_detection

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import android.view.TextureView
import android.widget.ImageView
import com.programminghut.pose_detection.ml.LiteModelMovenetSingleposeLightningTfliteFloat164
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.roundToInt
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {

    private val paint = Paint()
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var model: LiteModelMovenetSingleposeLightningTfliteFloat164
    private lateinit var bitmap: Bitmap
    private lateinit var imageView: ImageView
    private lateinit var handler: Handler
    private lateinit var handlerThread: HandlerThread
    private lateinit var textureView: TextureView
    private lateinit var cameraManager: CameraManager
    private val bodyPartNames = arrayOf("鼻子", "左眼", "右眼", "左耳", "右耳",
        "左肩", "右肩", "左肘", "右肘", "左腕",
        "右腕", "左胯", "右胯", "左膝", "右膝",
        "左踝", "右踝"
    )
    private val points: MutableList<Pair<Float, Float>> = mutableListOf() // Store screen coordinates of points
    private var isDetected = false
    private var angleCounter = 0
    private var prevAngle = 0f
    private var error = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        getPermissions()

        imageProcessor = ImageProcessor.Builder().add(ResizeOp(192, 192, ResizeOp.ResizeMethod.BILINEAR)).build()
        model = LiteModelMovenetSingleposeLightningTfliteFloat164.newInstance(this)
        imageView = findViewById(R.id.imageView)
        textureView = findViewById(R.id.textureView)
        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        handlerThread = HandlerThread("videoThread")
        handlerThread.start()
        handler = Handler(handlerThread.looper)

        paint.color = Color.YELLOW
        paint.textSize = 40f // Set text size for coordinates

        textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(p0: SurfaceTexture, p1: Int, p2: Int) {
                openCamera()
            }

            override fun onSurfaceTextureSizeChanged(p0: SurfaceTexture, p1: Int, p2: Int) {}

            override fun onSurfaceTextureDestroyed(p0: SurfaceTexture): Boolean { return false }

            @SuppressLint("UnsafeExperimentalUsageError")
            override fun onSurfaceTextureUpdated(p0: SurfaceTexture) {
                bitmap = textureView.bitmap!!
                var tensorImage = TensorImage(DataType.UINT8)
                tensorImage.load(bitmap)
                tensorImage = imageProcessor.process(tensorImage)

                val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 192, 192, 3), DataType.UINT8)
                inputFeature0.loadBuffer(tensorImage.buffer)

                val outputs = model.process(inputFeature0)
                val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

                points.clear() // Clear previous points

                var h = bitmap.height
                var w = bitmap.width
                var x = 0

                val mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                val canvas = Canvas(mutable) // Create canvas here

                while (x <= 49) {
                    val confidence = outputFeature0.get(x + 2)
                    if (confidence > 0.45) {
                        val bodyPartId = (x / 3)
                        if (bodyPartId == 5 || bodyPartId == 11 || bodyPartId == 13 || bodyPartId == 15) {
                            val pointX = outputFeature0.get(x + 1) * w
                            val pointY = outputFeature0.get(x) * h
                            points.add(pointX to pointY) // Store screen coordinates

                            val bodyPartName = bodyPartNames[bodyPartId]

                            // Draw point and label only if confidence is high enough
                            canvas.drawCircle(pointX, pointY, 10f, paint)
                            canvas.drawText(bodyPartName, pointX, pointY, paint) // Draw body part name

                            // Log the coordinates
                            Log.d("BodyPart", "$bodyPartName: ($pointX, $pointY)")
                        }
                    }
                    x += 3
                }

                // Connect points if both points are detected
                if (points.size >= 3) {
                    val angle = calculateAngle(points[0], points[1], points[2]) // Angle between points 5, 11, and 13
                    if (prevAngle > 60 && angle < 60) {
                        error++
                    } else if (prevAngle < 60 && angle > 60) {
                        angleCounter++
                    }
                    prevAngle = angle
                    val angleText = "Crunch Angle: ${angle.roundToInt()}°"
                    canvas.drawText(angleText, 50f, (bitmap.height - 40f), paint)
                    canvas.drawText("次数: $angleCounter", 50f, 280f, Paint(paint).apply {
                        color = Color.YELLOW // 设置文本颜色为黄色
                        textSize = 160f // 设置文本大小为40
                    })
                    canvas.drawText("Error: $error", 50f, (bitmap.height - 120f), paint)
                    connectPoints(canvas, points[0], points[1]) // Connect points 5 and 11
                    connectPoints(canvas, points[1], points[2]) // Connect points 11 and 13
                    // 检查是否需要显示错误消息
                    if (error > 0) {
                        val errorMessage = "Abdominal curls are not in place"
                        val textPaint = Paint(paint).apply {
                            color = Color.RED // 设置文本颜色为红色
                            textSize = 50f // 设置文本大小为30
                        }
                        val textWidth = textPaint.measureText(errorMessage) // 获取文本的宽度
                        val textX = (bitmap.width - textWidth) / 2 // 计算使文本居中对齐的起始横坐标
                        val textY = bitmap.height - 1580f // 设置文本的纵坐标
                        canvas.drawText(errorMessage, textX, textY, textPaint)
                    } else {
                        // 清除之前的错误消息
                        canvas.drawText("", 50f, (bitmap.height - 160f), paint)
                    }

                }
                // 在 onSurfaceTextureUpdated 方法中添加以下代码
                if (points.size >= 3) { // 如果至少检测到 3 个关键点
                    val angle13 = calculateAngle(points[0], points[1], points[2]) // 计算关键点11、13、15之间的夹角
                    val angleText13 = "Knee flexion angle = ${angle13.roundToInt()}°" // 构建角度文本
                    var textSize = 50f // 设置文本大小
                    val textPaint = Paint(paint).apply {
                        color = Color.BLUE // 设置文本颜色为蓝色
                        textSize = textSize // 设置文本大小
                    } // 复制并设置文本画笔的大小
                    val textBounds = Rect()
                    textPaint.getTextBounds(angleText13, 0, angleText13.length, textBounds)
                    val textWidth = textPaint.measureText(angleText13)
                    val textHeight = textBounds.height() // 获取文本的高度
                    val textX = 50f // 设置文本的横坐标
                    val textY = bitmap.height - 230f // 设置文本的纵坐标
                    canvas.drawText(angleText13, textX, textY, textPaint) // 在左下角显示夹角

                    // 如果角度大于160度
                    if (angle13 > 120) {
                        val errorText = "Knee flexion angle is too large, please adjust."
                        val textPaint = Paint(paint).apply {
                            color = Color.RED // 设置文本颜色为红色
                            textSize = 20f // 设置文本大小为24
                        }
                        val textWidth = textPaint.measureText(errorText) // 获取文本的宽度
                        val textX = (bitmap.width - textWidth) / 2 // 计算使文本居中对齐的起始横坐标
                        val textY = 600f // 设置文本的纵坐标
                        canvas.drawText(errorText, textX, textY, textPaint)
                    } else {
                        // 清除之前的文本
                        canvas.drawText("", (bitmap.width - 800f), 100f, paint)
                    }

                }
                imageView.setImageBitmap(mutable)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        model.close()
    }

    @SuppressLint("MissingPermission")
    fun openCamera() {
        cameraManager.openCamera(cameraManager.cameraIdList[0], object : CameraDevice.StateCallback() {
            override fun onOpened(p0: CameraDevice) {
                val captureRequest = p0.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                val surface = Surface(textureView.surfaceTexture)
                captureRequest.addTarget(surface)
                p0.createCaptureSession(listOf(surface), object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(p0: CameraCaptureSession) {
                        p0.setRepeatingRequest(captureRequest.build(), null, null)
                    }
                    override fun onConfigureFailed(p0: CameraCaptureSession) {}
                }, handler)
            }
            override fun onDisconnected(p0: CameraDevice) {}
            override fun onError(p0: CameraDevice, p1: Int) {}
        }, handler)
    }

    private fun connectPoints(canvas: Canvas, p1: Pair<Float, Float>, p2: Pair<Float, Float>) {
        canvas.drawLine(p1.first, p1.second, p2.first, p2.second, paint)
    }

    private fun calculateAngle(p1: Pair<Float, Float>, p2: Pair<Float, Float>, p3: Pair<Float, Float>): Float {
        val vector1 = Pair(p1.first - p2.first, p1.second - p2.second)
        val vector2 = Pair(p3.first - p2.first, p3.second - p2.second)
        val dotProduct = vector1.first * vector2.first + vector1.second * vector2.second
        val mag1 = sqrt((vector1.first * vector1.first + vector1.second * vector1.second).toDouble())
        val mag2 = sqrt((vector2.first * vector2.first + vector2.second * vector2.second).toDouble())
        val angle = Math.acos(dotProduct / (mag1 * mag2))
        return Math.toDegrees(angle).toFloat()
    }

    private fun getPermissions() {
        if (checkSelfPermission(android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 101)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED) getPermissions()
    }
}
