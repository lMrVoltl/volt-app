package com.voltage.ai

import android.annotation.SuppressLint
import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.content.pm.PackageManager
import android.hardware.camera2.CameraManager
import android.provider.AlarmClock
import android.provider.Settings
import android.util.Base64
import android.webkit.JavascriptInterface
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import okhttp3.Dns
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import android.speech.tts.TextToSpeech
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.util.Locale
import java.util.concurrent.CountDownLatch

import java.net.URL
import java.net.UnknownHostException
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {
    companion object {
        init { System.loadLibrary("volt_jni") }
        const val CATALOG_URL = "https://raw.githubusercontent.com/lMrVoltl/volt-models/main/models.json"
    }

    private lateinit var webView: WebView
    private var fileUploadCallback: ValueCallback<Array<Uri>>? = null

    private val fileChooserLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val uris = WebChromeClient.FileChooserParams.parseResult(result.resultCode, result.data)
        fileUploadCallback?.onReceiveValue(uris)
        fileUploadCallback = null
    }

    private val customModelPickerLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            thread {
                try {
                    val rawName = uri.lastPathSegment?.substringAfterLast('/') ?: "custom-model.gguf"
                    val cleanName = if (rawName.endsWith(".gguf", true)) rawName else "$rawName.gguf"
                    val id = "custom-" + cleanName.removeSuffix(".gguf").replace(Regex("[^a-zA-Z0-9._-]"), "_")
                    val target = File(modelsDir(), "$id.gguf")
                    contentResolver.openInputStream(uri)?.use { input ->
                        FileOutputStream(target).use { output ->
                            input.copyTo(output, 65536)
                        }
                    }
                    val sizeMb = target.length() / (1024 * 1024)
                    val obj = JSONObject()
                        .put("id", id)
                        .put("name", cleanName.removeSuffix(".gguf"))
                        .put("tier", "Локальная модель (.gguf)")
                        .put("base_model", cleanName)
                        .put("params", "Custom GGUF")
                        .put("size_mb", sizeMb)
                        .put("min_free_ram_mb", 1500)
                        .put("vision", false)
                        .put("uncensored", true)
                        .put("description", "Импортировано вручную с устройства.")
                        .put("download_url", "")
                        .put("mmproj_url", "")
                        .put("downloaded", true)
                    val quoted = JSONObject.quote(obj.toString())
                    runOnUiThread { webView.evaluateJavascript("window.onCustomModelImported($quoted)", null) }
                } catch (e: Exception) {
                    val err = JSONObject.quote(e.message ?: "Ошибка импорта файла")
                    runOnUiThread { webView.evaluateJavascript("window.onCustomModelError($err)", null) }
                }
            }
        }
    }

    private val smartDns = object : Dns {
        override fun lookup(hostname: String): List<InetAddress> {
            return try {
                Dns.SYSTEM.lookup(hostname)
            } catch (e: UnknownHostException) {
                val res = URL("https://8.8.8.8/resolve?name=$hostname&type=A").readText()
                val answers = JSONObject(res).optJSONArray("Answer") ?: throw e
                val ips = mutableListOf<InetAddress>()
                for (i in 0 until answers.length()) {
                    val item = answers.getJSONObject(i)
                    if (item.optInt("type") == 1) ips.add(InetAddress.getByName(item.getString("data")))
                }
                if (ips.isEmpty()) throw e else ips
            }
        }
    }

    private val client = OkHttpClient.Builder()
        .dns(smartDns)
        .followRedirects(true)
        .followSslRedirects(true)
        .connectTimeout(25, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .build()

    external fun nativeLoadModel(path: String, mmprojPath: String, threads: Int, nCtx: Int): Boolean
    external fun promptStartTimeMs = System.currentTimeMillis(); firstTokenTimeMs = 0L; nativeGenerate(prompt: String, sysPrompt: String, rgbBytes: ByteArray?, imgW: Int, imgH: Int, temp: Float, topP: Float, maxTokens: Int)
    external fun nativeStop()

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        webView = WebView(this).apply {
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            settings.allowFileAccess = true
            webViewClient = WebViewClient()
            webChromeClient = object : WebChromeClient() {
                override fun onShowFileChooser(
                    view: WebView?,
                    filePathCallback: ValueCallback<Array<Uri>>?,
                    fileChooserParams: FileChooserParams?
                ): Boolean {
                    fileUploadCallback?.onReceiveValue(null)
                    fileUploadCallback = filePathCallback
                    val intent = fileChooserParams?.createIntent() ?: return false
                    fileChooserLauncher.launch(intent)
                    return true
                }
            }
            addJavascriptInterface(VoltBridge(), "VoltAndroid")
            loadUrl("file:///android_asset/index.html")
        }
        setContentView(webView)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && !Environment.isExternalStorageManager()) {
            startActivity(Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, Uri.parse("package:$packageName")))
        }
    }

    private fun modelsDir(): File {
        val appDir = File(getExternalFilesDir(null) ?: filesDir, "VoltModels")
        if (!appDir.exists()) appDir.mkdirs()
        return appDir
    }

    private fun resolveModelFile(fileName: String): File {
        val appFile = File(modelsDir(), fileName)
        if (appFile.exists() && appFile.length() > 10_000_000L) return appFile
        val pubDir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "VoltModels")
        val pubFile = File(pubDir, fileName)
        if (pubFile.exists() && pubFile.length() > 10_000_000L) return pubFile
        return appFile
    }

    private fun downloadSingleFile(urlStr: String, target: File, progressStart: Int, progressSpan: Int, id: String) {
        val tmp = File(target.parentFile, target.name + ".part")
        val req = Request.Builder().url(urlStr).build()
        client.newCall(req).execute().use { resp ->
            if (!resp.isSuccessful) throw Exception("HTTP ${resp.code}")
            val body = resp.body ?: throw Exception("Нет данных")
            val total = body.contentLength()
            var downloaded = 0L
            var lastPct = -1

            body.byteStream().use { input ->
                FileOutputStream(tmp).use { output ->
                    val buf = ByteArray(65536)
                    while (true) {
                        val r = input.read(buf)
                        if (r <= 0) break
                        output.write(buf, 0, r)
                        downloaded += r
                        if (total > 0) {
                            val pct = progressStart + ((downloaded * progressSpan) / total).toInt()
                            if (pct != lastPct) {
                                lastPct = pct
                                runOnUiThread {
                                    webView.evaluateJavascript("window.onDownloadProgress('$id', $pct)", null)
                                }
                            }
                        }
                    }
                }
            }
        }
        if (target.exists()) target.delete()
        tmp.renameTo(target)
    }

    fun onNativeToken(token: String) {
        val escaped = JSONObject.quote(token)
        runOnUiThread { webView.evaluateJavascript("window.if (firstTokenTimeMs == 0L) firstTokenTimeMs = System.currentTimeMillis(); if (apiGenerationLatch != null) apiStreamBuffer.append(token); onVoltToken($escaped)", null) }
    }

    fun onNativeStats(tokens: Int, tps: Double, totalSec: Double, visionState: Int, visionErr: String) {
        runOnUiThread {
            val qErr = JSONObject.quote(visionErr)
            webView.evaluateJavascript("window.onVoltStats($tokens, $tps, $totalSec, $visionState, $qErr)", null)
        }
    }

    inner class VoltBridge {
        @JavascriptInterface
        fun getDeviceRamJson(): String {
            val am = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            val mi = ActivityManager.MemoryInfo()
            am.getMemoryInfo(mi)
            return JSONObject()
                .put("free_mb", mi.availMem / (1024 * 1024))
                .put("total_mb", mi.totalMem / (1024 * 1024))
                .toString()
        }

        @JavascriptInterface
        fun getDeviceHardwareJson(): String {
            val rawSoc = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL else Build.HARDWARE
            val socTitle = when {
                rawSoc.contains("7450", true) -> "Qualcomm Snapdragon 7 Gen 1"
                rawSoc.contains("7325", true) -> "Qualcomm Snapdragon 778G"
                rawSoc.contains("8475", true) -> "Qualcomm Snapdragon 8+ Gen 1"
                rawSoc.contains("8550", true) -> "Qualcomm Snapdragon 8 Gen 2"
                rawSoc.contains("8650", true) -> "Qualcomm Snapdragon 8 Gen 3"
                rawSoc.isNotBlank() && rawSoc != "unknown" -> rawSoc
                else -> Build.MODEL
            }

            val cpuInfo = try { File("/proc/cpuinfo").readText() } catch (_: Exception) { "" }
            val hasDotProd = cpuInfo.contains("asimddp", true)
            val hasFp16 = cpuInfo.contains("fphp", true) || cpuInfo.contains("asimdhp", true)
            val hasI8mm = cpuInfo.contains("i8mm", true)

            val simdList = mutableListOf("ARM64 NEON")
            if (hasDotProd) simdList.add("DotProd")
            if (hasFp16) simdList.add("FP16")
            if (hasI8mm) simdList.add("I8MM")

            val cores = Runtime.getRuntime().availableProcessors()
            val am = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            val mi = ActivityManager.MemoryInfo()
            am.getMemoryInfo(mi)
            val totalGb = Math.round((mi.totalMem / (1024.0 * 1024.0 * 1024.0)) * 10.0) / 10.0

            val supported = JSONArray().apply {
                put(JSONObject().put("tier", "Volt Ion v2.5 • 1.5B Fast").put("quant", "Qwen 2.5 1.5B • Текст и файлы").put("status", "Максимальная скорость"))
                put(JSONObject().put("tier", "Volt Neutron v2.5 • 1.5B Uncensored").put("quant", "Qwen 2.5 1.5B Abliterated • Без фильтров").put("status", "Свободный диалог"))
                put(JSONObject().put("tier", "Volt Photon v2.0 • 2B Vision").put("quant", "Qwen2-VL 2B + mmproj FP16").put("status", "Субфлагман (фото + текст)"))
                if (totalGb >= 7.5) {
                    put(JSONObject().put("tier", "Volt Plasma v2.5 • 3B Vision").put("quant", "Qwen2.5-VL 3B + mmproj Q8_0").put("status", if (hasDotProd) "Флагман (ускорение DotProd)" else "Флагман"))
                }
            }

            return JSONObject()
                .put("device", "${Build.MANUFACTURER.uppercase()} ${Build.MODEL}")
                .put("soc", socTitle)
                .put("socCode", rawSoc)
                .put("cores", cores)
                .put("totalGb", totalGb)
                .put("simd", simdList.joinToString(" • "))
                .put("supported", supported)
                .toString()
        }

        @JavascriptInterface
        fun recognizeImageOcr(base64Data: String) {
            try {
                val clean = base64Data.substringAfter(",")
                val bytes = Base64.decode(clean, Base64.DEFAULT)
                val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                val image = InputImage.fromBitmap(bmp, 0)
                val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
                recognizer.process(image)
                    .addOnSuccessListener { visionText ->
                        val q = JSONObject.quote(visionText.text)
                        webView.evaluateJavascript("window.onOcrDone($q)", null)
                    }
                    .addOnFailureListener {
                        webView.evaluateJavascript("window.onOcrDone('')", null)
                    }
            } catch (_: Exception) {
                runOnUiThread { webView.evaluateJavascript("window.onOcrDone('')", null) }
            }
        }

        @JavascriptInterface
        fun fetchCatalog() {
            thread {
                try {
                    val req = Request.Builder().url(CATALOG_URL).build()
                    val raw = client.newCall(req).execute().use { resp ->
                        if (!resp.isSuccessful) throw Exception("HTTP ${resp.code}")
                        resp.body?.string() ?: throw Exception("Пустой ответ")
                    }
                    val json = JSONObject(raw)
                    val models = json.getJSONArray("models")
                    val dir = modelsDir()
                    for (i in 0 until models.length()) {
                        val m = models.getJSONObject(i)
                        val id = m.getString("id")
                        val file = resolveModelFile("$id.gguf")
                        val hasMmproj = m.optString("mmproj_url", "").isNotEmpty()
                        val mmFile = resolveModelFile("$id-mmproj.gguf")
                        val ready = file.exists() && file.length() > 10_000_000 &&
                            (!hasMmproj || (mmFile.exists() && mmFile.length() > 10_000_000))
                        m.put("downloaded", ready)
                    }
                    val appDir = modelsDir()
                    appDir.listFiles()?.filter { it.isFile && it.name.startsWith("custom-") && it.name.endsWith(".gguf") && it.length() > 10_000_000L }?.forEach { cf ->
                        val cid = cf.name.removeSuffix(".gguf")
                        val cname = cid.removePrefix("custom-")
                        models.put(JSONObject()
                            .put("id", cid)
                            .put("name", cname)
                            .put("tier", "Локальная модель (.gguf)")
                            .put("base_model", cf.name)
                            .put("params", "Custom GGUF")
                            .put("size_mb", cf.length() / (1024 * 1024))
                            .put("min_free_ram_mb", 1500)
                            .put("vision", false)
                            .put("uncensored", true)
                            .put("description", "Локальный файл модели с устройства.")
                            .put("download_url", "")
                            .put("mmproj_url", "")
                            .put("downloaded", true))
                    }
                    val quoted = JSONObject.quote(json.toString())
                    runOnUiThread { webView.evaluateJavascript("window.onCatalogLoaded($quoted)", null) }
                } catch (e: Exception) {
                    val err = JSONObject.quote(e.message ?: "Ошибка сети")
                    runOnUiThread { webView.evaluateJavascript("window.onCatalogError($err)", null) }
                }
            }
        }

        @JavascriptInterface
        fun downloadModel(id: String, urlStr: String, mmprojUrl: String) {
            thread {
                try {
                    val dir = modelsDir()
                    val target = File(dir, "$id.gguf")
                    if (mmprojUrl.isNotEmpty()) {
                        downloadSingleFile(urlStr, target, 0, 65, id)
                        val mmTarget = File(dir, "$id-mmproj.gguf")
                        downloadSingleFile(mmprojUrl, mmTarget, 65, 35, id)
                    } else {
                        downloadSingleFile(urlStr, target, 0, 100, id)
                    }
                    runOnUiThread { webView.evaluateJavascript("window.onDownloadDone('$id')", null) }
                } catch (e: Exception) {
                    val err = JSONObject.quote(e.message ?: "Ошибка загрузки")
                    runOnUiThread { webView.evaluateJavascript("window.onDownloadError('$id', $err)", null) }
                }
            }
        }

        @JavascriptInterface
        fun loadModel(id: String, threads: Int, nCtx: Int) {
            thread {
                val file = resolveModelFile("$id.gguf")
                val mmFile = resolveModelFile("$id-mmproj.gguf")
                val mmPath = if (mmFile.exists() && mmFile.length() > 10_000_000) mmFile.absolutePath else ""
                val ok = file.exists() && nativeLoadModel(file.absolutePath, mmPath, threads, nCtx)
                runOnUiThread { webView.evaluateJavascript("window.onModelLoaded('$id', $ok)", null) }
            }
        }

        @JavascriptInterface
        fun sendPrompt(formattedHistory: String, sysPrompt: String, imgBase64: String, temp: Float, topP: Float, maxTokens: Int) {
            thread {
                var rgbBytes: ByteArray? = null
                var w = 0
                var h = 0
                if (imgBase64.isNotEmpty()) {
                    try {
                        val clean = imgBase64.substringAfter(",")
                        val bytes = Base64.decode(clean, Base64.DEFAULT)
                        val rawBmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                        val maxSide = 336
                        val scale = minOf(maxSide.toFloat() / rawBmp.width, maxSide.toFloat() / rawBmp.height, 1.0f)
                        val targetW = maxOf(28, (((rawBmp.width * scale).toInt() + 14) / 28) * 28)
                        val targetH = maxOf(28, (((rawBmp.height * scale).toInt() + 14) / 28) * 28)
                        val scaled = Bitmap.createScaledBitmap(rawBmp, targetW, targetH, true)
                        w = scaled.width
                        h = scaled.height
                        val pixels = IntArray(w * h)
                        scaled.getPixels(pixels, 0, w, 0, 0, w, h)
                        val rgb = ByteArray(w * h * 3)
                        for (i in pixels.indices) {
                            val px = pixels[i]
                            rgb[i * 3] = ((px shr 16) and 0xFF).toByte()
                            rgb[i * 3 + 1] = ((px shr 8) and 0xFF).toByte()
                            rgb[i * 3 + 2] = (px and 0xFF).toByte()
                        }
                        rgbBytes = rgb
                    } catch (_: Exception) {}
                }
                promptStartTimeMs = System.currentTimeMillis(); firstTokenTimeMs = 0L; nativeGenerate(formattedHistory, sysPrompt, rgbBytes, w, h, temp, topP, maxTokens)
                runOnUiThread { webView.evaluateJavascript("window.apiGenerationLatch?.countDown(); onVoltDone()", null) }
            }
        }

        @JavascriptInterface
        fun pickCustomModelFile() {
            runOnUiThread {
                customModelPickerLauncher.launch(arrayOf("*/*"))
            }
        }

        @JavascriptInterface
        fun isAgentAccessibilityEnabled(): Boolean = VoltAgentService.isEnabled()

        @JavascriptInterface
        fun getScreenUiState(): String = VoltAgentService.dumpScreenTree()

        @JavascriptInterface
        fun openAccessibilitySettings() {
            runOnUiThread {
                startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS).apply {
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                })
            }
        }

        @JavascriptInterface
        fun executeAgentAction(jsonAction: String): String {
            return try {
                val obj = JSONObject(jsonAction)
                val action = obj.optString("action", "").lowercase()
                val target = obj.optString("target", "")
                val value = obj.optString("value", "")

                when (action) {
                    "open_app" -> {
                        val pm = packageManager
                        val apps = pm.getInstalledApplications(PackageManager.GET_META_DATA)
                        var matchedPkg: String? = null
                        for (app in apps) {
                            val label = pm.getApplicationLabel(app).toString()
                            if (label.equals(target, true) || label.contains(target, true) || app.packageName.contains(target, true)) {
                                if (pm.getLaunchIntentForPackage(app.packageName) != null) {
                                    matchedPkg = app.packageName
                                    if (label.equals(target, true)) break
                                }
                            }
                        }
                        if (matchedPkg != null) {
                            val launch = pm.getLaunchIntentForPackage(matchedPkg)!!
                            launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                            startActivity(launch)
                            "Открыто приложение: $target"
                        } else {
                            "Приложение «$target» не найдено"
                        }
                    }
                    "set_alarm" -> {
                        val parts = target.split(":")
                        val hour = parts.getOrNull(0)?.trim()?.toIntOrNull() ?: 8
                        val min = parts.getOrNull(1)?.trim()?.toIntOrNull() ?: 0
                        val intent = Intent(AlarmClock.ACTION_SET_ALARM).apply {
                            putExtra(AlarmClock.EXTRA_HOUR, hour)
                            putExtra(AlarmClock.EXTRA_MINUTES, min)
                            putExtra(AlarmClock.EXTRA_MESSAGE, value.ifEmpty { "Volt AI" })
                            putExtra(AlarmClock.EXTRA_SKIP_UI, true)
                            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                        }
                        startActivity(intent)
                        "Будильник установлен на %02d:%02d".format(hour, min)
                    }
                    "set_timer" -> {
                        val sec = target.filter { it.isDigit() }.toIntOrNull() ?: 60
                        val intent = Intent(AlarmClock.ACTION_SET_TIMER).apply {
                            putExtra(AlarmClock.EXTRA_LENGTH, sec)
                            putExtra(AlarmClock.EXTRA_MESSAGE, value.ifEmpty { "Volt AI Таймер" })
                            putExtra(AlarmClock.EXTRA_SKIP_UI, true)
                            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                        }
                        startActivity(intent)
                        "Таймер запущен на $sec сек."
                    }
                    "flashlight" -> {
                        val cm = getSystemService(Context.CAMERA_SERVICE) as CameraManager
                        val camId = cm.cameraIdList.firstOrNull() ?: return "Вспышка не найдена"
                        val turnOn = target.equals("on", true) || target.equals("вкл", true) || target == "1"
                        cm.setTorchMode(camId, turnOn)
                        if (turnOn) "Фонарик включён" else "Фонарик выключен"
                    }
                    "search_web" -> {
                        val uri = Uri.parse("https://www.google.com/search?q=" + Uri.encode(target))
                        startActivity(Intent(Intent.ACTION_VIEW, uri).apply { addFlags(Intent.FLAG_ACTIVITY_NEW_TASK) })
                        "Открыт поиск: $target"
                    }
                    "open_settings" -> {
                        val act = when (target.lowercase()) {
                            "wifi" -> Settings.ACTION_WIFI_SETTINGS
                            "bluetooth" -> Settings.ACTION_BLUETOOTH_SETTINGS
                            "battery" -> Settings.ACTION_BATTERY_SAVER_SETTINGS
                            else -> Settings.ACTION_SETTINGS
                        }
                        startActivity(Intent(act).apply { addFlags(Intent.FLAG_ACTIVITY_NEW_TASK) })
                        "Открыты настройки ($target)"
                    }
                    "click_id" -> {
                        val idx = target.filter { it.isDigit() }.toIntOrNull() ?: -1
                        if (!VoltAgentService.isEnabled()) "Нужно включить спец. возможности для Volt AI Агента"
                        else if (VoltAgentService.clickById(idx)) "Нажат элемент [$idx]"
                        else "Элемент [$idx] не найден"
                    }
                    "tap_xy" -> {
                        val coords = target.split(",")
                        val x = coords.getOrNull(0)?.trim()?.toFloatOrNull() ?: 0f
                        val y = coords.getOrNull(1)?.trim()?.toFloatOrNull() ?: 0f
                        if (!VoltAgentService.isEnabled()) "Нужно включить спец. возможности"
                        else if (VoltAgentService.tapAt(x, y)) "Тап по ($x, $y)"
                        else "Не удалось выполнить тап"
                    }
                    "click" -> {
                        if (!VoltAgentService.isEnabled()) "Нужно включить спец. возможности для Volt AI Агента"
                        else if (VoltAgentService.clickByText(target)) "Нажато: «$target»"
                        else "Элемент «$target» не найден на экране"
                    }
                    "input_text" -> {
                        val idx = value.filter { it.isDigit() }.toIntOrNull() ?: -1
                        if (!VoltAgentService.isEnabled()) "Нужно включить спец. возможности для Volt AI Агента"
                        else if (VoltAgentService.inputText(target, idx)) "Введён текст: «$target»"
                        else "Поле ввода не найдено"
                    }
                    "scroll" -> {
                        if (!VoltAgentService.isEnabled()) "Нужно включить спец. возможности для Volt AI Агента"
                        else if (VoltAgentService.scrollDown()) "Экран прокручен вниз"
                        else "Не удалось прокрутить экран"
                    }
                    "back" -> {
                        VoltAgentService.goBack()
                        "Выполнено: Назад"
                    }
                    "home" -> {
                        VoltAgentService.goHome()
                        "Переход на главный экран"
                    }
                    else -> "Неизвестная команда агента: $action"
                }
            } catch (e: Exception) {
                "Ошибка выполнения: ${e.message}"
            }
        }

        @JavascriptInterface
        fun stopGeneration() { nativeStop() }
    }

    // === VOLT LOCAL OPENAI API SERVER (127.0.0.1:8080) & TTS ===
    private var localApiServer: ServerSocket? = null
    @Volatile private var isApiServerRunning = false
    @Volatile private var apiStreamBuffer = StringBuilder()
    @Volatile private var apiGenerationLatch: CountDownLatch? = null
    private var ttsEngine: TextToSpeech? = null
    @Volatile private var promptStartTimeMs: Long = 0L
    @Volatile private var firstTokenTimeMs: Long = 0L

    private fun initVoltTts() {
        if (ttsEngine == null) {
            ttsEngine = TextToSpeech(this) { status ->
                if (status == TextToSpeech.SUCCESS) {
                    ttsEngine?.language = Locale("ru", "RU")
                }
            }
        }
    }

    @JavascriptInterface
    fun speakText(text: String) {
        initVoltTts()
        val clean = text.replace(Regex("[*#`>]"), "").trim()
        ttsEngine?.speak(clean, TextToSpeech.QUEUE_FLUSH, null, "volt_tts")
    }

    @JavascriptInterface
    fun stopSpeaking() {
        ttsEngine?.stop()
    }

    @JavascriptInterface
    fun getLastTtftMs(): Long {
        return if (firstTokenTimeMs > promptStartTimeMs && promptStartTimeMs > 0L) {
            firstTokenTimeMs - promptStartTimeMs
        } else 0L
    }

    @JavascriptInterface
    fun isLocalServerRunning(): Boolean = isApiServerRunning

    @JavascriptInterface
    fun toggleLocalApiServer(enable: Boolean, port: Int = 8080): String {
        if (!enable) {
            isApiServerRunning = false
            try { localApiServer?.close() } catch (_: Exception) {}
            localApiServer = null
            return "STOPPED"
        }
        if (isApiServerRunning) return "RUNNING:http://127.0.0.1:$port/v1"
        isApiServerRunning = true
        thread(name = "VoltOpenAIServer") {
            try {
                val srv = ServerSocket(port, 16, InetAddress.getByName("127.0.0.1"))
                localApiServer = srv
                while (isApiServerRunning) {
                    val client = try { srv.accept() } catch (_: Exception) { break }
                    thread { handleOpenAiClient(client) }
                }
            } catch (e: Exception) {
                isApiServerRunning = false
            }
        }
        return "RUNNING:http://127.0.0.1:$port/v1"
    }

    private fun handleOpenAiClient(socket: Socket) {
        try {
            socket.soTimeout = 120000
            val reader = BufferedReader(InputStreamReader(socket.getInputStream(), "UTF-8"))
            val requestLine = reader.readLine() ?: return
            var contentLength = 0
            var line: String?
            while (reader.readLine().also { line = it } != null && line!!.isNotEmpty()) {
                if (line!!.lowercase().startsWith("content-length:")) {
                    contentLength = line!!.substringAfter(":").trim().toIntOrNull() ?: 0
                }
            }
            var body = ""
            if (contentLength > 0) {
                val buf = CharArray(contentLength)
                var readTotal = 0
                while (readTotal < contentLength) {
                    val r = reader.read(buf, readTotal, contentLength - readTotal)
                    if (r <= 0) break
                    readTotal += r
                }
                body = String(buf, 0, readTotal)
            }

            val out = OutputStreamWriter(socket.getOutputStream(), "UTF-8")
            if (requestLine.startsWith("OPTIONS ")) {
                out.write("HTTP/1.1 204 No Content
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: *

")
                out.flush()
                return
            }

            if (requestLine.contains("/v1/models")) {
                val modelsJson = "{"object":"list","data":[{"id":"volt-local","object":"model","owned_by":"VOLTAGE"}]}"
                out.write("HTTP/1.1 200 OK
Content-Type: application/json; charset=utf-8
Access-Control-Allow-Origin: *
Content-Length: ${modelsJson.toByteArray().size}

$modelsJson")
                out.flush()
                return
            }

            if (requestLine.contains("/v1/chat/completions") && body.isNotEmpty()) {
                val reqObj = JSONObject(body)
                val msgs = reqObj.optJSONArray("messages") ?: JSONArray()
                val temp = reqObj.optDouble("temperature", 0.7).toFloat()
                val topP = reqObj.optDouble("top_p", 0.9).toFloat()
                val maxTok = reqObj.optInt("max_tokens", 512)
                var sysPrompt = "Ты — полезный AI-ассистент Volt AI."
                val chatSb = StringBuilder()
                for (i in 0 until msgs.length()) {
                    val m = msgs.getJSONObject(i)
                    val role = m.optString("role", "user")
                    val content = m.optString("content", "")
                    if (role == "system") sysPrompt = content
                    else chatSb.append("<|im_start|>").append(role).append("
").append(content).append("<|im_end|>
")
                }
                chatSb.append("<|im_start|>assistant
")

                synchronized(this) {
                    apiStreamBuffer = StringBuilder()
                    val latch = CountDownLatch(1)
                    apiGenerationLatch = latch
                    sendPrompt(chatSb.toString(), sysPrompt, "", temp, topP, maxTok)
                    latch.await(110, TimeUnit.SECONDS)
                    apiGenerationLatch = null
                }

                val replyText = apiStreamBuffer.toString().trim()
                val respObj = JSONObject().apply {
                    put("id", "chatcmpl-volt-${System.currentTimeMillis()}")
                    put("object", "chat.completion")
                    put("created", System.currentTimeMillis() / 1000)
                    put("model", "volt-local")
                    put("choices", JSONArray().put(JSONObject().apply {
                        put("index", 0)
                        put("message", JSONObject().apply {
                            put("role", "assistant")
                            put("content", replyText)
                        })
                        put("finish_reason", "stop")
                    }))
                }
                val respStr = respObj.toString()
                out.write("HTTP/1.1 200 OK
Content-Type: application/json; charset=utf-8
Access-Control-Allow-Origin: *
Content-Length: ${respStr.toByteArray().size}

$respStr")
                out.flush()
            } else {
                val err = "{"error":"Unsupported endpoint"}"
                out.write("HTTP/1.1 404 Not Found
Content-Type: application/json
Content-Length: ${err.length}

$err")
                out.flush()
            }
        } catch (_: Exception) {
        } finally {
            try { socket.close() } catch (_: Exception) {}
        }
    }

}
