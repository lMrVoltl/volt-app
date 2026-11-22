package com.voltage.ai

import android.annotation.SuppressLint
import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.hardware.camera2.CameraManager
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.AlarmClock
import android.provider.Settings
import android.speech.tts.TextToSpeech
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
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.net.Socket
import java.net.URL
import java.net.UnknownHostException
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {
    companion object {
        init { System.loadLibrary("volt_jni") }
        const val CATALOG_URL = "https://raw.githubusercontent.com/lMrVoltl/volt-models/main/models.json"
        const val UPSCALER_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        const val UPSCALER_FILE = "RealESRGAN_x4plus.pth"
    }

    private lateinit var webView: WebView
    private var fileUploadCallback: ValueCallback<Array<Uri>>? = null
    private var pendingAttachTargetId: String? = null
    @Volatile private var isModelLoaded = false

    private var localApiServer: ServerSocket? = null
    @Volatile private var isApiServerRunning = false
    @Volatile private var apiStreamBuffer = StringBuilder()
    @Volatile private var apiGenerationLatch: CountDownLatch? = null
    @Volatile private var activeSseOut: OutputStream? = null
    @Volatile private var activeSseId: String = ""
    @Volatile private var activeSseModel: String = "volt-local"
    @Volatile private var activeSseCreated: Long = 0L
    private var ttsEngine: TextToSpeech? = null
    @Volatile private var promptStartTimeMs: Long = 0L
    @Volatile private var firstTokenTimeMs: Long = 0L

    @Volatile private var sdProcess: Process? = null
    @Volatile private var sdStoppedByUser = false

    private val attachTargetLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val targetId = pendingAttachTargetId
        pendingAttachTargetId = null
        val uri = result.data?.data ?: return@registerForActivityResult
        if (targetId.isNullOrEmpty()) return@registerForActivityResult
        thread {
            try {
                val dest = File(modelsDir(), "$targetId.gguf")
                if (dest.exists()) dest.delete()
                contentResolver.openInputStream(uri)?.use { inp ->
                    FileOutputStream(dest).use { out ->
                        val buf = ByteArray(262144)
                        while (true) {
                            val r = inp.read(buf)
                            if (r <= 0) break
                            out.write(buf, 0, r)
                        }
                    }
                }
                runOnUiThread {
                    webView.evaluateJavascript("window.onModelAttached('$targetId', true, '')", null)
                }
            } catch (e: Exception) {
                val err = JSONObject.quote(e.message ?: "Ошибка чтения файла")
                runOnUiThread {
                    webView.evaluateJavascript("window.onModelAttached('$targetId', false, $err)", null)
                }
            }
        }
    }

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

    private val sdModelPickerLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            thread {
                try {
                    val rawName = uri.lastPathSegment?.substringAfterLast('/')?.substringAfterLast(':') ?: "custom_sd.safetensors"
                    val ext = when {
                        rawName.endsWith(".gguf", true) -> ".gguf"
                        rawName.endsWith(".ckpt", true) -> ".ckpt"
                        else -> ".safetensors"
                    }
                    val base = rawName.substringBeforeLast('.').replace(Regex("[^a-zA-Z0-9._-]"), "_").ifBlank { "custom_sd" }
                    val target = File(sdModelsDir(), "$base$ext")
                    if (target.exists()) target.delete()
                    contentResolver.openInputStream(uri)?.use { input ->
                        FileOutputStream(target).use { output ->
                            val buf = ByteArray(262144)
                            while (true) {
                                val r = input.read(buf)
                                if (r <= 0) break
                                output.write(buf, 0, r)
                            }
                        }
                    }
                    runOnUiThread { webView.evaluateJavascript("window.onSdModelsUpdated()", null) }
                } catch (e: Exception) {
                    val err = JSONObject.quote(e.message ?: "Ошибка импорта SD-модели")
                    runOnUiThread { webView.evaluateJavascript("window.onSdError($err)", null) }
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
    external fun nativeGenerate(prompt: String, sysPrompt: String, rgbBytes: ByteArray?, imgW: Int, imgH: Int, temp: Float, topP: Float, maxTokens: Int)
    external fun nativeStop()

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        workspaceDir()
        sdModelsDir()
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

    override fun onDestroy() {
        isApiServerRunning = false
        try { localApiServer?.close() } catch (_: Exception) {}
        try { sdProcess?.destroyForcibly() } catch (_: Exception) {}
        ttsEngine?.shutdown()
        super.onDestroy()
    }

    private fun modelsDir(): File {
        val appDir = File(getExternalFilesDir(null) ?: filesDir, "VoltModels")
        if (!appDir.exists()) appDir.mkdirs()
        return appDir
    }

    private fun sdModelsDir(): File {
        val d = File(modelsDir(), "sd")
        if (!d.exists()) d.mkdirs()
        return d
    }

    private fun upscalerFile(): File = File(sdModelsDir(), UPSCALER_FILE)

    private fun studioOutputDir(): File {
        val pics = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
        val dir = File(pics, "VoltStudio")
        if (!dir.exists()) dir.mkdirs()
        return dir
    }

    private fun workspaceDir(): File {
        val dl = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        val ws = File(dl, "VoltWorkspace")
        if (!ws.exists()) ws.mkdirs()
        return ws
    }

    private fun resolveWorkspaceFile(relOrAbs: String): File {
        val clean = relOrAbs.trim()
        if (clean.isEmpty() || clean == ".") return workspaceDir()
        val direct = File(clean)
        return if (direct.isAbsolute) direct else File(workspaceDir(), clean)
    }

    private fun resolveModelFile(fileName: String): File {
        val appFile = File(modelsDir(), fileName)
        if (appFile.exists() && appFile.length() > 10_000_000L) return appFile
        val pubDir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "VoltModels")
        val pubFile = File(pubDir, fileName)
        if (pubFile.exists() && pubFile.length() > 10_000_000L) return pubFile
        return appFile
    }

    private fun resolveSdBinary(): File = File(applicationInfo.nativeLibraryDir, "libvolt_sd.so")

    private fun configureSdEnv(pb: ProcessBuilder, useGpu: Boolean) {
        if (!useGpu) {
            pb.environment()["GGML_VK_VISIBLE_DEVICES"] = ""
        }
    }

    private fun runSdOnce(
        cmd: List<String>,
        useGpu: Boolean,
        maxGenPct: Int,
        quantType: String
    ): Triple<Boolean, Boolean, String> {
        val modeTag = if (useGpu) "Vulkan GPU" else "CPU"
        runOnUiThread {
            webView.evaluateJavascript("window.onSdProgress(4, 'Загрузка весов ($modeTag • $quantType)...')", null)
        }
        val pb = ProcessBuilder(cmd).directory(cacheDir).redirectErrorStream(true)
        configureSdEnv(pb, useGpu)
        val proc = pb.start()
        sdProcess = proc

        val logTail = ArrayDeque<String>()
        val stepRegex = Regex("(\\d+)\\s*/\\s*(\\d+)")
        val lineBuf = ByteArrayOutputStream()
        val stream = proc.inputStream
        var detectedVulkan = false

        while (true) {
            val b = stream.read()
            if (b == -1) break
            if (b == '\n'.code || b == '\r'.code) {
                if (lineBuf.size() > 0) {
                    val line = lineBuf.toString("UTF-8").trim()
                    lineBuf.reset()
                    if (line.isNotEmpty()) {
                        if (logTail.size >= 14) logTail.removeFirst()
                        logTail.addLast(line)
                        if (line.contains("Vulkan0", true) || line.contains("ggml_vulkan", true) || line.contains("Adreno", true)) {
                            if (useGpu) detectedVulkan = true
                        }
                        val mStep = stepRegex.find(line)
                        when {
                            mStep != null && (line.contains("|") || line.contains("it/s", true) || line.contains("s/it", true)) -> {
                                val cur = mStep.groupValues[1].toIntOrNull() ?: 0
                                val tot = mStep.groupValues[2].toIntOrNull() ?: 0
                                if (tot > 45 && cur <= tot) {
                                    val pct = 5 + ((cur * 13) / tot)
                                    val msg = JSONObject.quote("Загрузка весов: $cur / $tot...")
                                    runOnUiThread { webView.evaluateJavascript("window.onSdProgress($pct, $msg)", null) }
                                } else if (tot in 1..45 && cur <= tot) {
                                    val pct = 20 + ((cur * (maxGenPct - 20)) / tot)
                                    val tag = if (detectedVulkan) "⚡ GPU" else "CPU"
                                    val msg = JSONObject.quote("$tag • Шаг $cur / $tot...")
                                    runOnUiThread { webView.evaluateJavascript("window.onSdProgress($pct, $msg)", null) }
                                }
                            }
                            line.contains("decoding", true) || line.contains("decode_first_stage", true) -> {
                                val vaePct = maxGenPct + 2
                                runOnUiThread { webView.evaluateJavascript("window.onSdProgress($vaePct, 'Декодирование VAE...')", null) }
                            }
                        }
                    }
                }
            } else {
                lineBuf.write(b)
            }
        }

        val exitCode = proc.waitFor()
        sdProcess = null
        val errTxt = logTail.takeLast(4).joinToString("\n").ifBlank { "Код выхода: $exitCode" }
        return Triple(exitCode == 0, detectedVulkan, errTxt)
    }

    private fun runUpscaleProcess(
        inputFile: File,
        outputFile: File,
        threads: Int,
        basePct: Int,
        useGpu: Boolean
    ): Triple<Boolean, Boolean, String> {
        val bin = resolveSdBinary()
        val upModel = upscalerFile()
        if (!bin.exists()) return Triple(false, false, "Движок libvolt_sd.so не найден")
        if (!upModel.exists() || upModel.length() < 10_000_000L) {
            return Triple(false, false, "Сначала скачай модель апскейлера Real-ESRGAN (64 МБ)")
        }

        val cmd = listOf(
            bin.absolutePath,
            "-M", "upscale",
            "--upscale-model", upModel.absolutePath,
            "-i", inputFile.absolutePath,
            "-o", outputFile.absolutePath,
            "-t", threads.toString()
        )
        val tag = if (useGpu) "Vulkan GPU" else "CPU"
        runOnUiThread {
            webView.evaluateJavascript("window.onSdProgress($basePct, 'Апскейл 4× ($tag)...')", null)
        }
        val pb = ProcessBuilder(cmd).directory(cacheDir).redirectErrorStream(true)
        configureSdEnv(pb, useGpu)
        val proc = pb.start()
        sdProcess = proc
        val tail = ArrayDeque<String>()
        val buf = ByteArrayOutputStream()
        val stream = proc.inputStream
        val pctRegex = Regex("(\\d+)%")
        var detectedVk = false

        while (true) {
            val b = stream.read()
            if (b == -1) break
            if (b == '\n'.code || b == '\r'.code) {
                if (buf.size() > 0) {
                    val line = buf.toString("UTF-8").trim()
                    buf.reset()
                    if (line.isNotEmpty()) {
                        if (tail.size >= 10) tail.removeFirst()
                        tail.addLast(line)
                        if (useGpu && (line.contains("Vulkan0", true) || line.contains("Adreno", true))) {
                            detectedVk = true
                        }
                        val mp = pctRegex.find(line)
                        if (mp != null) {
                            val rawP = mp.groupValues[1].toIntOrNull() ?: 0
                            val mapped = (basePct + ((rawP * (99 - basePct)) / 100)).coerceIn(basePct, 99)
                            val msg = JSONObject.quote("Апскейл 4×: $rawP%...")
                            runOnUiThread { webView.evaluateJavascript("window.onSdProgress($mapped, $msg)", null) }
                        }
                    }
                }
            } else {
                buf.write(b)
            }
        }
        val code = proc.waitFor()
        sdProcess = null
        return if (code == 0 && outputFile.exists() && outputFile.length() > 1000L) {
            Triple(true, detectedVk, "")
        } else {
            Triple(false, detectedVk, tail.takeLast(4).joinToString("\n").ifBlank { "Ошибка апскейла (код $code)" })
        }
    }

    private fun translitRuToEn(input: String): String {
        val map = mapOf(
            'а' to "a", 'б' to "b", 'в' to "v", 'г' to "g", 'д' to "d",
            'е' to "e", 'ё' to "e", 'ж' to "zh", 'з' to "z", 'и' to "i",
            'й' to "y", 'к' to "k", 'л' to "l", 'м' to "m", 'н' to "n",
            'о' to "o", 'п' to "p", 'р' to "r", 'с' to "s", 'т' to "t",
            'у' to "u", 'ф' to "f", 'х' to "h", 'ц' to "ts", 'ч' to "ch",
            'ш' to "sh", 'щ' to "shch", 'ъ' to "", 'ы' to "y", 'ь' to "",
            'э' to "e", 'ю' to "yu", 'я' to "ya"
        )
        val sb = StringBuilder()
        for (ch in input.lowercase()) sb.append(map[ch] ?: ch)
        return sb.toString()
    }

    private fun levenshtein(a: String, b: String): Int {
        if (a == b) return 0
        if (a.isEmpty()) return b.length
        if (b.isEmpty()) return a.length
        var prev = IntArray(b.length + 1) { it }
        var curr = IntArray(b.length + 1)
        for (i in 1..a.length) {
            curr[0] = i
            for (j in 1..b.length) {
                val cost = if (a[i - 1] == b[j - 1]) 0 else 1
                curr[j] = minOf(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            }
            val tmp = prev; prev = curr; curr = tmp
        }
        return prev[b.length]
    }

    private fun getLaunchableApps(): List<Pair<String, String>> {
        val pm = packageManager
        val mainIntent = Intent(Intent.ACTION_MAIN, null).apply { addCategory(Intent.CATEGORY_LAUNCHER) }
        val resolved = pm.queryIntentActivities(mainIntent, 0)
        return resolved.mapNotNull { ri ->
            val pkg = ri.activityInfo?.packageName ?: return@mapNotNull null
            if (pkg == packageName) return@mapNotNull null
            val label = ri.loadLabel(pm)?.toString()?.trim().orEmpty()
            if (label.isEmpty()) null else Pair(label, pkg)
        }.distinctBy { it.second }.sortedBy { it.first.lowercase() }
    }

    private fun findBestMatchingApp(query: String): Pair<String, String>? {
        val q = query.trim().lowercase()
        if (q.isEmpty()) return null
        val qTrans = translitRuToEn(q).replace(Regex("[^a-z0-9]"), "")
        val apps = getLaunchableApps()

        apps.firstOrNull { it.first.lowercase() == q }?.let { return it }
        apps.firstOrNull { it.first.lowercase().contains(q) || it.second.lowercase().contains(q) }?.let { return it }
        if (qTrans.length >= 2) {
            apps.firstOrNull { (label, pkg) ->
                val lTrans = translitRuToEn(label).replace(Regex("[^a-z0-9]"), "")
                val pNorm = pkg.lowercase().replace(Regex("[^a-z0-9]"), "")
                lTrans.contains(qTrans) || (qTrans.contains(lTrans) && lTrans.length >= 3) || pNorm.contains(qTrans)
            }?.let { return it }

            var best: Pair<String, String>? = null
            var bestScore = Double.MAX_VALUE
            for (app in apps) {
                val lTrans = translitRuToEn(app.first).replace(Regex("[^a-z0-9]"), "")
                if (lTrans.isEmpty()) continue
                val dist = levenshtein(qTrans, lTrans)
                val ratio = dist.toDouble() / maxOf(qTrans.length, lTrans.length)
                if (ratio < bestScore && ratio <= 0.45) {
                    bestScore = ratio
                    best = app
                }
            }
            if (best != null) return best
        }
        return null
    }

    private fun downloadSingleFile(
        urlStr: String, target: File, progressStart: Int, progressSpan: Int,
        id: String, jsProgressFn: String = "window.onDownloadProgress"
    ) {
        val tmp = File(target.parentFile, target.name + ".part")
        var downloaded = if (tmp.exists()) tmp.length() else 0L
        var lastPct = -1
        var attempt = 0
        val maxAttempts = 12

        val dlClient = client.newBuilder()
            .connectTimeout(20, TimeUnit.SECONDS)
            .readTimeout(45, TimeUnit.SECONDS)
            .writeTimeout(45, TimeUnit.SECONDS)
            .callTimeout(0, TimeUnit.SECONDS)
            .followRedirects(true)
            .followSslRedirects(true)
            .build()

        while (attempt < maxAttempts) {
            attempt++
            try {
                downloaded = if (tmp.exists()) tmp.length() else 0L
                val reqBuilder = Request.Builder()
                    .url(urlStr)
                    .header("User-Agent", "Mozilla/5.0 (Linux; Android 14) VoltAI/3.0")
                if (downloaded > 0L) reqBuilder.header("Range", "bytes=$downloaded-")

                dlClient.newCall(reqBuilder.build()).execute().use { resp ->
                    if (resp.code == 416) {
                        if (target.exists()) target.delete()
                        tmp.renameTo(target)
                        return
                    }
                    if (!resp.isSuccessful && resp.code != 206) throw Exception("HTTP ${resp.code}")

                    val appendMode = (resp.code == 206 && downloaded > 0L)
                    if (!appendMode) downloaded = 0L
                    val contentLen = resp.body?.contentLength() ?: -1L
                    val total = if (contentLen > 0L) downloaded + contentLen else -1L
                    val input = resp.body?.byteStream() ?: throw Exception("Пустой поток")

                    FileOutputStream(tmp, appendMode).use { output ->
                        val buf = ByteArray(131072)
                        while (true) {
                            val r = input.read(buf)
                            if (r <= 0) break
                            output.write(buf, 0, r)
                            downloaded += r
                            if (total > 0L) {
                                val pct = progressStart + ((downloaded * progressSpan) / total).toInt()
                                if (pct != lastPct) {
                                    lastPct = pct
                                    runOnUiThread {
                                        webView.evaluateJavascript("$jsProgressFn('$id', $pct)", null)
                                    }
                                }
                            }
                        }
                        output.flush()
                    }
                    if (total > 0L && downloaded < total) throw Exception("Неполная загрузка: $downloaded / $total")
                }

                if (target.exists()) target.delete()
                tmp.renameTo(target)
                return
            } catch (e: Exception) {
                if (e.message?.contains("HTTP 404") == true || e.message?.contains("HTTP 403") == true) throw e
                if (attempt >= maxAttempts) throw e
                try { Thread.sleep(1500L) } catch (_: Exception) {}
            }
        }
    }

    fun onNativeToken(token: String) {
        if (firstTokenTimeMs == 0L) firstTokenTimeMs = System.currentTimeMillis()
        if (apiGenerationLatch != null) {
            apiStreamBuffer.append(token)
            val sseOut = activeSseOut
            if (sseOut != null) {
                try {
                    val chunk = JSONObject().apply {
                        put("id", activeSseId)
                        put("object", "chat.completion.chunk")
                        put("created", activeSseCreated)
                        put("model", activeSseModel)
                        put("choices", JSONArray().put(JSONObject().apply {
                            put("index", 0)
                            put("delta", JSONObject().put("content", token))
                            put("finish_reason", JSONObject.NULL)
                        }))
                    }.toString()
                    sseOut.write("data: $chunk\n\n".toByteArray(Charsets.UTF_8))
                    sseOut.flush()
                } catch (_: Exception) {
                    activeSseOut = null
                    nativeStop()
                }
            }
            return
        }
        val escaped = JSONObject.quote(token)
        runOnUiThread { webView.evaluateJavascript("window.onVoltToken($escaped)", null) }
    }

    fun onNativeStats(tokens: Int, tps: Double, totalSec: Double, visionState: Int, visionErr: String) {
        if (apiGenerationLatch != null) return
        runOnUiThread {
            val qErr = JSONObject.quote(visionErr)
            webView.evaluateJavascript("window.onVoltStats($tokens, $tps, $totalSec, $visionState, $qErr)", null)
        }
    }

    private fun initVoltTts() {
        if (ttsEngine == null) {
            ttsEngine = TextToSpeech(this) { status ->
                if (status == TextToSpeech.SUCCESS) ttsEngine?.language = Locale("ru", "RU")
            }
        }
    }

    private fun startOrStopLocalServer(enable: Boolean, port: Int): String {
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
                val srv = ServerSocket()
                srv.reuseAddress = true
                srv.bind(InetSocketAddress(InetAddress.getByName("127.0.0.1"), port), 16)
                localApiServer = srv
                while (isApiServerRunning) {
                    val client = try { srv.accept() } catch (_: Exception) { break }
                    thread { handleOpenAiClient(client) }
                }
            } catch (_: Exception) {
                isApiServerRunning = false
            }
        }
        return "RUNNING:http://127.0.0.1:$port/v1"
    }

    private fun extractMessageText(contentAny: Any?): String {
        if (contentAny == null) return ""
        if (contentAny is String) return contentAny
        if (contentAny is JSONArray) {
            val sb = StringBuilder()
            for (i in 0 until contentAny.length()) {
                val item = contentAny.optJSONObject(i) ?: continue
                if (item.optString("type") == "text") {
                    if (sb.isNotEmpty()) sb.append("\n")
                    sb.append(item.optString("text", ""))
                }
            }
            return sb.toString()
        }
        return contentAny.toString()
    }

    private fun readHttpHeaderBytes(inp: InputStream): ByteArray {
        val baos = ByteArrayOutputStream()
        var state = 0
        while (true) {
            val b = inp.read()
            if (b == -1) break
            baos.write(b)
            state = when {
                state == 0 && b == '\r'.code -> 1
                state == 1 && b == '\n'.code -> 2
                state == 2 && b == '\r'.code -> 3
                state == 3 && b == '\n'.code -> return baos.toByteArray()
                b == '\r'.code -> 1
                else -> 0
            }
            if (baos.size() > 65536) break
        }
        return baos.toByteArray()
    }

    private fun readExactBytes(inp: InputStream, len: Int): ByteArray {
        val buf = ByteArray(len)
        var readTotal = 0
        while (readTotal < len) {
            val r = inp.read(buf, readTotal, len - readTotal)
            if (r <= 0) break
            readTotal += r
        }
        return if (readTotal == len) buf else buf.copyOf(readTotal)
    }

    private fun readChunkedBytes(inp: InputStream): ByteArray {
        val out = ByteArrayOutputStream()
        while (true) {
            val lineBaos = ByteArrayOutputStream()
            while (true) {
                val b = inp.read()
                if (b == -1 || b == '\n'.code) break
                if (b != '\r'.code) lineBaos.write(b)
            }
            val sizeStr = lineBaos.toString("UTF-8").trim().substringBefore(";")
            if (sizeStr.isEmpty()) break
            val chunkSize = sizeStr.toIntOrNull(16) ?: break
            if (chunkSize <= 0) break
            out.write(readExactBytes(inp, chunkSize))
            inp.read()
            inp.read()
        }
        return out.toByteArray()
    }

    private fun writeHttpJson(socket: Socket, status: String, json: String) {
        val bodyBytes = json.toByteArray(Charsets.UTF_8)
        val headers = "HTTP/1.1 $status\r\n" +
            "Content-Type: application/json; charset=utf-8\r\n" +
            "Access-Control-Allow-Origin: *\r\n" +
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n" +
            "Access-Control-Allow-Headers: *\r\n" +
            "Connection: close\r\n" +
            "Content-Length: ${bodyBytes.size}\r\n\r\n"
        val out = socket.getOutputStream()
        out.write(headers.toByteArray(Charsets.UTF_8))
        out.write(bodyBytes)
        out.flush()
    }

    private fun handleOpenAiClient(socket: Socket) {
        try {
            socket.soTimeout = 180000
            val inp = socket.getInputStream()
            val headerBytes = readHttpHeaderBytes(inp)
            if (headerBytes.isEmpty()) return

            val headerLines = String(headerBytes, Charsets.UTF_8).split("\r\n")
            val requestLine = headerLines.firstOrNull()?.trim().orEmpty()
            if (requestLine.isEmpty()) return

            var contentLength = 0
            var isChunked = false
            for (i in 1 until headerLines.size) {
                val h = headerLines[i]
                val hl = h.lowercase()
                if (hl.startsWith("content-length:")) {
                    contentLength = h.substringAfter(":").trim().toIntOrNull() ?: 0
                } else if (hl.startsWith("transfer-encoding:") && hl.contains("chunked")) {
                    isChunked = true
                }
            }

            val bodyBytes = when {
                isChunked -> readChunkedBytes(inp)
                contentLength > 0 -> readExactBytes(inp, contentLength)
                else -> ByteArray(0)
            }
            val body = String(bodyBytes, Charsets.UTF_8)

            if (requestLine.startsWith("OPTIONS ")) {
                val headers = "HTTP/1.1 204 No Content\r\n" +
                    "Access-Control-Allow-Origin: *\r\n" +
                    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n" +
                    "Access-Control-Allow-Headers: *\r\n" +
                    "Connection: close\r\n\r\n"
                socket.getOutputStream().write(headers.toByteArray(Charsets.UTF_8))
                socket.getOutputStream().flush()
                return
            }

            val path = requestLine.split(" ").getOrNull(1)?.substringBefore("?") ?: "/"

            if (requestLine.startsWith("GET ") && (path == "/" || path.endsWith("/models") || path == "/v1" || path == "/v1/")) {
                val modelsJson = JSONObject().apply {
                    put("object", "list")
                    put("data", JSONArray().put(JSONObject().apply {
                        put("id", "volt-local")
                        put("object", "model")
                        put("created", System.currentTimeMillis() / 1000)
                        put("owned_by", "VOLTAGE")
                    }))
                }.toString()
                writeHttpJson(socket, "200 OK", modelsJson)
                return
            }

            if (path.contains("chat/completions") && body.isNotEmpty()) {
                val reqObj = JSONObject(body)
                val msgs = reqObj.optJSONArray("messages") ?: JSONArray()
                val reqModel = reqObj.optString("model", "volt-local").ifBlank { "volt-local" }
                val isStream = reqObj.optBoolean("stream", false)
                val temp = reqObj.optDouble("temperature", 0.6).toFloat()
                val topP = reqObj.optDouble("top_p", 0.9).toFloat()
                val maxTok = reqObj.optInt("max_tokens", 512).coerceIn(32, 2048)

                var sysPrompt = "Ты — локальный AI-ассистент Volt AI (VOLTAGE). Отвечай чётко и по делу."
                val chatSb = StringBuilder()
                for (i in 0 until msgs.length()) {
                    val m = msgs.getJSONObject(i)
                    val role = m.optString("role", "user")
                    val content = extractMessageText(m.opt("content"))
                    if (role == "system") sysPrompt = content
                    else chatSb.append("<|im_start|>").append(role).append("\n").append(content).append("<|im_end|>\n")
                }

                val nowSec = System.currentTimeMillis() / 1000
                val compId = "chatcmpl-volt-${System.currentTimeMillis()}"

                if (!isModelLoaded) {
                    val errText = "Volt AI подключён, но модель ещё не загружена в RAM. Открой Volt AI и выбери модель сверху."
                    if (isStream) {
                        val out = socket.getOutputStream()
                        val sseHeaders = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n"
                        out.write(sseHeaders.toByteArray(Charsets.UTF_8))
                        val c1 = JSONObject().put("id", compId).put("object", "chat.completion.chunk").put("created", nowSec).put("model", reqModel)
                            .put("choices", JSONArray().put(JSONObject().put("index", 0).put("delta", JSONObject().put("role", "assistant").put("content", errText)).put("finish_reason", "stop"))).toString()
                        out.write("data: $c1\n\ndata: [DONE]\n\n".toByteArray(Charsets.UTF_8))
                        out.flush()
                    } else {
                        val respStr = JSONObject().put("id", compId).put("object", "chat.completion").put("created", nowSec).put("model", reqModel)
                            .put("choices", JSONArray().put(JSONObject().put("index", 0).put("message", JSONObject().put("role", "assistant").put("content", errText)).put("finish_reason", "stop"))).toString()
                        writeHttpJson(socket, "200 OK", respStr)
                    }
                    return
                }

                if (isStream) {
                    val out = socket.getOutputStream()
                    val sseHeaders = "HTTP/1.1 200 OK\r\n" +
                        "Content-Type: text/event-stream; charset=utf-8\r\n" +
                        "Cache-Control: no-cache\r\n" +
                        "Access-Control-Allow-Origin: *\r\n" +
                        "Connection: close\r\n\r\n"
                    out.write(sseHeaders.toByteArray(Charsets.UTF_8))
                    val roleChunk = JSONObject().apply {
                        put("id", compId)
                        put("object", "chat.completion.chunk")
                        put("created", nowSec)
                        put("model", reqModel)
                        put("choices", JSONArray().put(JSONObject().apply {
                            put("index", 0)
                            put("delta", JSONObject().put("role", "assistant"))
                            put("finish_reason", JSONObject.NULL)
                        }))
                    }.toString()
                    out.write("data: $roleChunk\n\n".toByteArray(Charsets.UTF_8))
                    out.flush()

                    synchronized(this) {
                        apiStreamBuffer = StringBuilder()
                        apiGenerationLatch = CountDownLatch(1)
                        activeSseId = compId
                        activeSseModel = reqModel
                        activeSseCreated = nowSec
                        activeSseOut = out
                        promptStartTimeMs = System.currentTimeMillis()
                        firstTokenTimeMs = 0L
                        try {
                            nativeGenerate(chatSb.toString(), sysPrompt, null, 0, 0, temp, topP, maxTok)
                        } finally {
                            activeSseOut = null
                            apiGenerationLatch?.countDown()
                            apiGenerationLatch = null
                        }
                    }

                    try {
                        val endChunk = JSONObject().apply {
                            put("id", compId)
                            put("object", "chat.completion.chunk")
                            put("created", nowSec)
                            put("model", reqModel)
                            put("choices", JSONArray().put(JSONObject().apply {
                                put("index", 0)
                                put("delta", JSONObject())
                                put("finish_reason", "stop")
                            }))
                        }.toString()
                        out.write("data: $endChunk\n\ndata: [DONE]\n\n".toByteArray(Charsets.UTF_8))
                        out.flush()
                    } catch (_: Exception) {}
                } else {
                    synchronized(this) {
                        apiStreamBuffer = StringBuilder()
                        apiGenerationLatch = CountDownLatch(1)
                        activeSseOut = null
                        promptStartTimeMs = System.currentTimeMillis()
                        firstTokenTimeMs = 0L
                        try {
                            nativeGenerate(chatSb.toString(), sysPrompt, null, 0, 0, temp, topP, maxTok)
                        } finally {
                            apiGenerationLatch?.countDown()
                            apiGenerationLatch = null
                        }
                    }
                    val replyText = apiStreamBuffer.toString().trim().ifEmpty { "Готово." }
                    val respStr = JSONObject().apply {
                        put("id", compId)
                        put("object", "chat.completion")
                        put("created", nowSec)
                        put("model", reqModel)
                        put("choices", JSONArray().put(JSONObject().apply {
                            put("index", 0)
                            put("message", JSONObject().apply {
                                put("role", "assistant")
                                put("content", replyText)
                            })
                            put("finish_reason", "stop")
                        }))
                        put("usage", JSONObject().apply {
                            put("prompt_tokens", chatSb.length / 4)
                            put("completion_tokens", replyText.length / 4)
                            put("total_tokens", (chatSb.length + replyText.length) / 4)
                        })
                    }.toString()
                    writeHttpJson(socket, "200 OK", respStr)
                }
            } else {
                writeHttpJson(socket, "404 Not Found", JSONObject().put("error", "Unsupported endpoint: $path").toString())
            }
        } catch (_: Exception) {
        } finally {
            try { socket.close() } catch (_: Exception) {}
        }
    }

    private fun saveBase64PngToFile(dataUrl: String, target: File): Boolean {
        return try {
            val clean = dataUrl.substringAfter(",")
            val bytes = Base64.decode(clean, Base64.DEFAULT)
            FileOutputStream(target).use { it.write(bytes) }
            true
        } catch (_: Exception) {
            false
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

            val simdList = mutableListOf("ARM64 NEON", "Vulkan GPU")
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
                put(JSONObject().put("tier", "Volt Neutron v2.6 • 3B Vision").put("quant", "Qwen2.5-VL 3B • Фото + текст").put("status", "Оптимум"))
                put(JSONObject().put("tier", "Volt Studio • SD + Real-ESRGAN 4×").put("quant", "stable-diffusion.cpp (Vulkan GPU + KleidiAI)").put("status", "txt2img / img2img / inpaint / 4x"))
                if (totalGb >= 7.5) {
                    put(JSONObject().put("tier", "Volt Plasma v3.0 • 8B").put("quant", "Qwen 3 8B Q4_K_M").put("status", if (hasDotProd) "Флагман (DotProd)" else "Флагман"))
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
        fun getInstalledAppsList(): String {
            return try {
                getLaunchableApps().take(70).joinToString(", ") { it.first }
            } catch (_: Exception) {
                ""
            }
        }

        // === VOLT STUDIO (STABLE DIFFUSION + VULKAN GPU + REAL-ESRGAN 4X) ===
        @JavascriptInterface
        fun getSdModelsJson(): String {
            val sdDir = sdModelsDir()
            val binOk = resolveSdBinary().exists()
            val upReady = upscalerFile().exists() && upscalerFile().length() > 10_000_000L

            val builtIns = listOf(
                mapOf(
                    "id" to "sd-turbo",
                    "file" to "sd-turbo.safetensors",
                    "name" to "SD-Turbo (Быстрая)",
                    "desc" to "4 шага, CFG 1.0 • Максимальная скорость на GPU/CPU + отлично с апскейлом 4×",
                    "size_mb" to 1710,
                    "steps" to 4,
                    "cfg" to 1.0,
                    "url" to "https://huggingface.co/stabilityai/sd-turbo/resolve/main/sd_turbo.safetensors"
                ),
                mapOf(
                    "id" to "dreamshaper-8",
                    "file" to "dreamshaper-8.safetensors",
                    "name" to "DreamShaper 8 (Без цензуры)",
                    "desc" to "10 шагов, CFG 7.0 • Реализм, арт, улучшение фото и маска (inpainting)",
                    "size_mb" to 2130,
                    "steps" to 10,
                    "cfg" to 7.0,
                    "url" to "https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors"
                )
            )

            val arr = JSONArray()
            val knownFiles = HashSet<String>()
            knownFiles.add(UPSCALER_FILE)
            for (b in builtIns) {
                val fname = b["file"] as String
                knownFiles.add(fname)
                val f = File(sdDir, fname)
                val downloaded = f.exists() && f.length() > 100_000_000L
                arr.put(JSONObject()
                    .put("id", b["id"])
                    .put("name", b["name"])
                    .put("desc", b["desc"])
                    .put("size_mb", b["size_mb"])
                    .put("default_steps", b["steps"])
                    .put("default_cfg", b["cfg"])
                    .put("url", b["url"])
                    .put("path", f.absolutePath)
                    .put("downloaded", downloaded)
                    .put("custom", false))
            }

            sdDir.listFiles()?.filter {
                !knownFiles.contains(it.name) && it.length() > 50_000_000L &&
                    (it.name.endsWith(".safetensors", true) || it.name.endsWith(".ckpt", true) || it.name.endsWith(".gguf", true))
            }?.sortedBy { it.name }?.forEach { f ->
                arr.put(JSONObject()
                    .put("id", "custom-" + f.nameWithoutExtension)
                    .put("name", f.nameWithoutExtension)
                    .put("desc", "Локальный чекпоинт (${f.extension.uppercase()}) • ${f.length() / (1024 * 1024)} МБ")
                    .put("size_mb", f.length() / (1024 * 1024))
                    .put("default_steps", 10)
                    .put("default_cfg", 7.0)
                    .put("url", "")
                    .put("path", f.absolutePath)
                    .put("downloaded", true)
                    .put("custom", true))
            }

            return JSONObject()
                .put("engine_ready", binOk)
                .put("upscaler_ready", upReady)
                .put("models", arr)
                .toString()
        }

        @JavascriptInterface
        fun downloadUpscaler() {
            thread {
                try {
                    downloadSingleFile(UPSCALER_URL, upscalerFile(), 0, 100, "upscaler", "window.onUpscalerProgress")
                    runOnUiThread { webView.evaluateJavascript("window.onUpscalerDone()", null) }
                } catch (e: Exception) {
                    val err = JSONObject.quote(e.message ?: "Ошибка загрузки Real-ESRGAN")
                    runOnUiThread { webView.evaluateJavascript("window.onUpscalerError($err)", null) }
                }
            }
        }

        @JavascriptInterface
        fun downloadSdModel(id: String, urlStr: String) {
            thread {
                try {
                    val target = File(sdModelsDir(), "$id.safetensors")
                    downloadSingleFile(urlStr, target, 0, 100, id, "window.onSdDownloadProgress")
                    runOnUiThread { webView.evaluateJavascript("window.onSdDownloadDone('$id')", null) }
                } catch (e: Exception) {
                    val err = JSONObject.quote(e.message ?: "Ошибка загрузки SD-модели")
                    runOnUiThread { webView.evaluateJavascript("window.onSdDownloadError('$id', $err)", null) }
                }
            }
        }

        @JavascriptInterface
        fun deleteSdModel(path: String): Boolean {
            return try {
                val f = File(path)
                if (f.exists()) f.delete() else false
            } catch (_: Exception) {
                false
            }
        }

        @JavascriptInterface
        fun importSdModelFile() {
            runOnUiThread { sdModelPickerLauncher.launch(arrayOf("*/*")) }
        }

        @JavascriptInterface
        fun stopSdGeneration() {
            sdStoppedByUser = true
            try { sdProcess?.destroyForcibly() } catch (_: Exception) {}
            sdProcess = null
        }

        @JavascriptInterface
        fun upscaleImage(paramsJson: String) {
            thread(name = "VoltUpscaleWorker") {
                sdStoppedByUser = false
                val tStart = System.currentTimeMillis()
                try {
                    val p = JSONObject(paramsJson)
                    val imgB64 = p.optString("image", "")
                    val threads = p.optInt("threads", 4).coerceIn(1, 6)
                    val useGpu = p.optBoolean("useGpu", true)
                    if (imgB64.isEmpty()) {
                        runOnUiThread { webView.evaluateJavascript("window.onSdError('Выбери фото для апскейла')", null) }
                        return@thread
                    }
                    val inFile = File(cacheDir, "sd_up_in_${System.currentTimeMillis()}.png")
                    val outFile = File(cacheDir, "sd_up_out_${System.currentTimeMillis()}.png")
                    if (!saveBase64PngToFile(imgB64, inFile)) {
                        runOnUiThread { webView.evaluateJavascript("window.onSdError('Не удалось прочитать изображение')", null) }
                        return@thread
                    }

                    var (ok, usedVk, errMsg) = runUpscaleProcess(inFile, outFile, threads, 10, useGpu)
                    if (!ok && useGpu && !sdStoppedByUser) {
                        val retry = runUpscaleProcess(inFile, outFile, threads, 10, false)
                        ok = retry.first; usedVk = false; errMsg = retry.third
                    }
                    inFile.delete()

                    if (sdStoppedByUser) {
                        outFile.delete()
                        runOnUiThread { webView.evaluateJavascript("window.onSdError('Апскейл остановлен')", null) }
                        return@thread
                    }

                    if (ok) {
                        val savedFile = File(studioOutputDir(), "Volt_Up4x_${System.currentTimeMillis()}.png")
                        outFile.copyTo(savedFile, overwrite = true)
                        MediaScannerConnection.scanFile(this@MainActivity, arrayOf(savedFile.absolutePath), arrayOf("image/png"), null)
                        val dataUrl = "data:image/png;base64," + Base64.encodeToString(outFile.readBytes(), Base64.NO_WRAP)
                        outFile.delete()
                        val upSec = (System.currentTimeMillis() - tStart) / 1000.0
                        val backendLabel = if (usedVk) "⚡ Vulkan GPU" else "CPU"
                        val qUrl = JSONObject.quote(dataUrl)
                        val qPath = JSONObject.quote(savedFile.absolutePath)
                        val qBack = JSONObject.quote(backendLabel)
                        runOnUiThread { webView.evaluateJavascript("window.onSdDone($qUrl, $qPath, 0.0, $upSec, $qBack)", null) }
                    } else {
                        outFile.delete()
                        val qErr = JSONObject.quote(errMsg)
                        runOnUiThread { webView.evaluateJavascript("window.onSdError($qErr)", null) }
                    }
                } catch (e: Exception) {
                    val qErr = JSONObject.quote(e.message ?: "Ошибка апскейла")
                    runOnUiThread { webView.evaluateJavascript("window.onSdError($qErr)", null) }
                }
            }
        }

        @JavascriptInterface
        fun generateImage(paramsJson: String) {
            thread(name = "VoltSdWorker") {
                sdStoppedByUser = false
                val tGenStart = System.currentTimeMillis()
                try {
                    val bin = resolveSdBinary()
                    if (!bin.exists()) {
                        val err = JSONObject.quote("Движок libvolt_sd.so не найден в APK.")
                        runOnUiThread { webView.evaluateJavascript("window.onSdError($err)", null) }
                        return@thread
                    }

                    val p = JSONObject(paramsJson)
                    val modelPath = p.optString("modelPath", "")
                    val modelFile = File(modelPath)
                    if (!modelFile.exists()) {
                        val err = JSONObject.quote("Сначала скачай или импортируй SD-модель.")
                        runOnUiThread { webView.evaluateJavascript("window.onSdError($err)", null) }
                        return@thread
                    }

                    val mode = p.optString("mode", "txt2img")
                    val prompt = p.optString("prompt", "").trim()
                    val negPrompt = p.optString("negPrompt", "").trim()
                    val width = maxOf(256, (p.optInt("width", 320) / 64) * 64)
                    val height = maxOf(256, (p.optInt("height", 320) / 64) * 64)
                    val steps = p.optInt("steps", 8).coerceIn(1, 40)
                    val cfgScale = p.optDouble("cfgScale", 1.0).coerceIn(0.5, 15.0)
                    val strength = p.optDouble("strength", 0.55).coerceIn(0.1, 0.99)
                    val seed = p.optLong("seed", -1L)
                    val threads = p.optInt("threads", 4).coerceIn(1, 6)
                    val quantType = p.optString("quantType", "q8_0")
                    val useGpu = p.optBoolean("useGpu", true)
                    val autoUpscale = p.optBoolean("autoUpscale", false)
                    val initB64 = p.optString("initImage", "")
                    val maskB64 = p.optString("maskImage", "")

                    val outFile = File(cacheDir, "sd_out_${System.currentTimeMillis()}.png")
                    val cmd = mutableListOf(
                        bin.absolutePath,
                        "-m", modelFile.absolutePath,
                        "-p", prompt,
                        "-W", width.toString(),
                        "-H", height.toString(),
                        "--steps", steps.toString(),
                        "--cfg-scale", String.format(Locale.US, "%.2f", cfgScale),
                        "--sampling-method", "euler_a",
                        "-t", threads.toString(),
                        "-s", seed.toString(),
                        "--vae-tiling",
                        "--diffusion-fa",
                        "-o", outFile.absolutePath
                    )
                    if (quantType == "q8_0" || quantType == "q4_0" || quantType == "f16") {
                        cmd.add("--type")
                        cmd.add(quantType)
                    }
                    if (negPrompt.isNotEmpty()) {
                        cmd.add("-n")
                        cmd.add(negPrompt)
                    }

                    if ((mode == "img2img" || mode == "inpaint") && initB64.isNotEmpty()) {
                        val initFile = File(cacheDir, "sd_init.png")
                        if (saveBase64PngToFile(initB64, initFile)) {
                            cmd.add("-i")
                            cmd.add(initFile.absolutePath)
                            cmd.add("--strength")
                            cmd.add(String.format(Locale.US, "%.2f", strength))
                        }
                    }
                    if (mode == "inpaint" && maskB64.isNotEmpty()) {
                        val maskFile = File(cacheDir, "sd_mask.png")
                        if (saveBase64PngToFile(maskB64, maskFile)) {
                            cmd.add("--mask")
                            cmd.add(maskFile.absolutePath)
                        }
                    }

                    val maxGenPct = if (autoUpscale) 84 else 94
                    var (ok, usedVk, errTxt) = runSdOnce(cmd, useGpu, maxGenPct, quantType)

                    // Авто-откат на CPU, если Vulkan GPU драйвер упал
                    if (!ok && useGpu && !sdStoppedByUser) {
                        runOnUiThread {
                            webView.evaluateJavascript("window.onSdProgress(6, 'Vulkan недоступен — переключение на CPU...')", null)
                        }
                        val retry = runSdOnce(cmd, false, maxGenPct, quantType)
                        ok = retry.first; usedVk = false; errTxt = retry.third
                    }

                    if (sdStoppedByUser) {
                        outFile.delete()
                        runOnUiThread { webView.evaluateJavascript("window.onSdError('Генерация остановлена')", null) }
                        return@thread
                    }

                    val genSec = (System.currentTimeMillis() - tGenStart) / 1000.0

                    if (ok && outFile.exists() && outFile.length() > 1000L) {
                        var finalFile = outFile
                        var upSec = 0.0

                        if (autoUpscale && upscalerFile().exists() && upscalerFile().length() > 10_000_000L) {
                            val tUpStart = System.currentTimeMillis()
                            val upOut = File(cacheDir, "sd_up_${System.currentTimeMillis()}.png")
                            var (upOk, _, _) = runUpscaleProcess(outFile, upOut, threads, 88, usedVk)
                            if (!upOk && usedVk && !sdStoppedByUser) {
                                val upRetry = runUpscaleProcess(outFile, upOut, threads, 88, false)
                                upOk = upRetry.first
                            }
                            if (upOk) {
                                outFile.delete()
                                finalFile = upOut
                                upSec = (System.currentTimeMillis() - tUpStart) / 1000.0
                            } else {
                                upOut.delete()
                            }
                        }

                        if (sdStoppedByUser) {
                            finalFile.delete()
                            runOnUiThread { webView.evaluateJavascript("window.onSdError('Остановлено пользователем')", null) }
                            return@thread
                        }

                        val savedFile = File(studioOutputDir(), "Volt_${System.currentTimeMillis()}.png")
                        finalFile.copyTo(savedFile, overwrite = true)
                        MediaScannerConnection.scanFile(this@MainActivity, arrayOf(savedFile.absolutePath), arrayOf("image/png"), null)

                        val pngBytes = finalFile.readBytes()
                        val dataUrl = "data:image/png;base64," + Base64.encodeToString(pngBytes, Base64.NO_WRAP)
                        finalFile.delete()

                        val backendLabel = if (usedVk) "⚡ Vulkan GPU" else "CPU"
                        val qUrl = JSONObject.quote(dataUrl)
                        val qPath = JSONObject.quote(savedFile.absolutePath)
                        val qBack = JSONObject.quote(backendLabel)
                        runOnUiThread {
                            webView.evaluateJavascript("window.onSdDone($qUrl, $qPath, $genSec, $upSec, $qBack)", null)
                        }
                    } else {
                        val qErr = JSONObject.quote(errTxt)
                        runOnUiThread { webView.evaluateJavascript("window.onSdError($qErr)", null) }
                    }
                } catch (e: Exception) {
                    sdProcess = null
                    val qErr = JSONObject.quote(e.message ?: "Ошибка генерации изображения")
                    runOnUiThread { webView.evaluateJavascript("window.onSdError($qErr)", null) }
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
        fun stopSpeaking() { ttsEngine?.stop() }

        @JavascriptInterface
        fun getLastTtftMs(): Long {
            return if (firstTokenTimeMs > promptStartTimeMs && promptStartTimeMs > 0L) {
                firstTokenTimeMs - promptStartTimeMs
            } else 0L
        }

        @JavascriptInterface
        fun isLocalServerRunning(): Boolean = isApiServerRunning

        @JavascriptInterface
        fun toggleLocalApiServer(enable: Boolean, port: Int): String {
            return startOrStopLocalServer(enable, if (port > 0) port else 8080)
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

        private fun getFallbackCatalogJson(): String {
            return """{
              "models": [
                {
                  "id": "volt-photon-2.0",
                  "family": "Volt Photon",
                  "version": "v2.0 (Быстрая)",
                  "family_desc": "Сверхлёгкая карманная модель для мгновенных ответов без нагрузки на батарею",
                  "name": "Volt Photon v2.0",
                  "base_model": "Qwen2.5-1.5B-Instruct (Q4_K_M)",
                  "params": "1.5B",
                  "size_mb": 1120,
                  "min_free_ram_mb": 1800,
                  "vision": false,
                  "description": "Молниеносный запуск и ответы в оффлайне с минимальным расходом RAM",
                  "download_url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
                  "mmproj_url": ""
                },
                {
                  "id": "volt-neutron-2.5",
                  "family": "Volt Neutron",
                  "version": "v2.5 (Qwen 2.5)",
                  "family_desc": "Универсальный мультимодальный ИИ на базе Qwen 2.5 для фото, текста и учёбы",
                  "name": "Volt Neutron v2.5",
                  "base_model": "Qwen2.5-3B-Instruct (Q4_K_M)",
                  "params": "3B",
                  "size_mb": 2100,
                  "min_free_ram_mb": 2800,
                  "vision": false,
                  "description": "Классическая текстовая версия Qwen 2.5 3B",
                  "download_url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
                  "mmproj_url": ""
                },
                {
                  "id": "volt-neutron-2.6",
                  "family": "Volt Neutron",
                  "version": "v2.6 (Qwen 2.5-VL • Улучшенная)",
                  "family_desc": "Универсальный мультимодальный ИИ на базе Qwen 2.5 для фото, текста и учёбы",
                  "name": "Volt Neutron v2.6",
                  "base_model": "Qwen2.5-VL-3B-Instruct (Q4_K_M)",
                  "params": "3B + Vision",
                  "size_mb": 2580,
                  "min_free_ram_mb": 3000,
                  "vision": true,
                  "description": "Полноценная Q4_K_M версия Qwen 2.5-VL: точный анализ фотографий, конспектов и связный диалог",
                  "download_url": "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
                  "mmproj_url": "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"
                },
                {
                  "id": "volt-plasma-2.5",
                  "family": "Volt Plasma",
                  "version": "v2.5 (Qwen 2.5-VL 7B)",
                  "family_desc": "Флагманский интеллект Volt AI: максимальная глубина рассуждений, сложный код и анализ",
                  "name": "Volt Plasma v2.5",
                  "base_model": "Qwen2.5-VL-7B-Instruct (Q4_K_M)",
                  "params": "7B + Vision",
                  "size_mb": 5450,
                  "min_free_ram_mb": 4600,
                  "vision": true,
                  "description": "Мультимодальный флагман 7B с поддержкой распознавания сложных фотографий",
                  "download_url": "https://huggingface.co/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                  "mmproj_url": "https://huggingface.co/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf"
                },
                {
                  "id": "volt-plasma-3.0",
                  "family": "Volt Plasma",
                  "version": "v3.0 (Qwen 3 • 8B)",
                  "family_desc": "Флагманский интеллект Volt AI: максимальная глубина рассуждений, сложный код и анализ",
                  "name": "Volt Plasma v3.0",
                  "base_model": "Qwen3-8B (Q4_K_M)",
                  "params": "8B (Qwen 3)",
                  "size_mb": 5030,
                  "min_free_ram_mb": 4800,
                  "vision": false,
                  "description": "Старший флагман поколения Qwen 3 (8B): глубокая логика, математика, код и точные ответы",
                  "download_url": "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
                  "mmproj_url": ""
                }
              ]
            }"""
        }

        private fun buildCatalogPayload(rawJsonStr: String, isOnline: Boolean): String {
            val json = JSONObject(rawJsonStr)
            json.put("online", isOnline)
            val models = json.getJSONArray("models")
            val dlRoot = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
            val scanDirs = listOfNotNull(modelsDir(), dlRoot)

            for (i in 0 until models.length()) {
                val m = models.getJSONObject(i)
                val id = m.getString("id")
                val dlUrl = m.optString("download_url", "")
                val origFileName = dlUrl.substringAfterLast("/").substringBefore("?")
                var file = resolveModelFile("$id.gguf")
                if (!file.exists() && origFileName.endsWith(".gguf")) {
                    for (d in scanDirs) {
                        val cand = File(d, origFileName)
                        if (cand.exists() && cand.length() > 50_000_000L) {
                            file = cand
                            break
                        }
                    }
                }
                val hasMmproj = m.optString("mmproj_url", "").isNotEmpty()
                val origMmName = m.optString("mmproj_url", "").substringAfterLast("/").substringBefore("?")
                var mmFile = resolveModelFile("$id-mmproj.gguf")
                if (hasMmproj && !mmFile.exists() && origMmName.endsWith(".gguf")) {
                    for (d in scanDirs) {
                        val cand = File(d, origMmName)
                        if (cand.exists() && cand.length() > 10_000_000L) {
                            mmFile = cand
                            break
                        }
                    }
                }
                val ok = file.exists() && file.length() > 50_000_000L &&
                        (!hasMmproj || (mmFile.exists() && mmFile.length() > 10_000_000L))
                m.put("downloaded", ok)
                if (file.exists()) m.put("resolved_path", file.absolutePath)
                if (mmFile.exists()) m.put("resolved_mmproj_path", mmFile.absolutePath)
            }
            val dir = modelsDir()
            dir.listFiles()?.filter { it.name.startsWith("custom-") && it.name.endsWith(".gguf") }?.forEach { f ->
                val cid = f.nameWithoutExtension
                models.put(JSONObject()
                    .put("id", cid)
                    .put("family", f.nameWithoutExtension.removePrefix("custom-"))
                    .put("version", "Локальный GGUF")
                    .put("family_desc", "Импортированная пользователем модель")
                    .put("name", f.nameWithoutExtension.removePrefix("custom-"))
                    .put("base_model", "Custom GGUF")
                    .put("params", "${f.length() / (1024 * 1024)} МБ")
                    .put("size_mb", f.length() / (1024 * 1024))
                    .put("min_free_ram_mb", 1500)
                    .put("vision", false)
                    .put("description", "Локальный файл из памяти устройства")
                    .put("downloaded", true))
            }
            return json.toString()
        }

        @JavascriptInterface
        fun fetchCatalog() {
            thread {
                val cacheFile = File(filesDir, "models_cache.json")
                try {
                    val localRaw = if (cacheFile.exists()) cacheFile.readText() else getFallbackCatalogJson()
                    val instantPayload = buildCatalogPayload(localRaw, false)
                    val qInstant = JSONObject.quote(instantPayload)
                    runOnUiThread { webView.evaluateJavascript("window.onCatalogLoaded($qInstant)", null) }
                } catch (_: Exception) {}

                try {
                    val fastClient = client.newBuilder()
                        .connectTimeout(4, TimeUnit.SECONDS)
                        .readTimeout(4, TimeUnit.SECONDS)
                        .callTimeout(5, TimeUnit.SECONDS)
                        .build()
                    val req = Request.Builder().url(CATALOG_URL).build()
                    val netRaw = fastClient.newCall(req).execute().use { resp ->
                        if (!resp.isSuccessful) throw Exception("HTTP ${resp.code}")
                        resp.body?.string() ?: throw Exception("Empty")
                    }
                    val netObj = JSONObject(netRaw)
                    val netArr = netObj.getJSONArray("models")
                    val fbArr = JSONObject(getFallbackCatalogJson()).getJSONArray("models")
                    val existingIds = HashSet<String>()
                    for (i in 0 until netArr.length()) existingIds.add(netArr.getJSONObject(i).getString("id"))
                    for (i in 0 until fbArr.length()) {
                        val fbItem = fbArr.getJSONObject(i)
                        if (!existingIds.contains(fbItem.getString("id"))) netArr.put(fbItem)
                    }
                    val mergedStr = netObj.toString()
                    cacheFile.writeText(mergedStr)
                    val onlinePayload = buildCatalogPayload(mergedStr, true)
                    val qOnline = JSONObject.quote(onlinePayload)
                    runOnUiThread { webView.evaluateJavascript("window.onCatalogLoaded($qOnline)", null) }
                } catch (_: Exception) {}
            }
        }

        @JavascriptInterface
        fun deleteModel(id: String): Boolean {
            return try {
                val f1 = resolveModelFile("$id.gguf")
                val f2 = resolveModelFile("$id-mmproj.gguf")
                val d1 = File(modelsDir(), "$id.gguf")
                val d2 = File(modelsDir(), "$id-mmproj.gguf")
                if (f1.exists()) f1.delete()
                if (f2.exists()) f2.delete()
                if (d1.exists()) d1.delete()
                if (d2.exists()) d2.delete()
                true
            } catch (_: Exception) {
                false
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
                val dlRoot = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                val scanDirs = listOfNotNull(modelsDir(), dlRoot)
                var file = resolveModelFile("$id.gguf")
                var mmFile = resolveModelFile("$id-mmproj.gguf")

                try {
                    val fbArr = JSONObject(getFallbackCatalogJson()).getJSONArray("models")
                    for (i in 0 until fbArr.length()) {
                        val item = fbArr.getJSONObject(i)
                        if (item.getString("id") == id) {
                            val origName = item.optString("download_url", "").substringAfterLast("/").substringBefore("?")
                            val origMm = item.optString("mmproj_url", "").substringAfterLast("/").substringBefore("?")
                            if (!file.exists() && origName.endsWith(".gguf")) {
                                scanDirs.firstOrNull { File(it, origName).exists() }?.let { file = File(it, origName) }
                            }
                            if (!mmFile.exists() && origMm.endsWith(".gguf")) {
                                scanDirs.firstOrNull { File(it, origMm).exists() }?.let { mmFile = File(it, origMm) }
                            }
                        }
                    }
                } catch (_: Exception) {}

                val mmPath = if (mmFile.exists() && mmFile.length() > 10_000_000) mmFile.absolutePath else ""
                val ok = file.exists() && nativeLoadModel(file.absolutePath, mmPath, threads, nCtx)
                isModelLoaded = ok
                runOnUiThread { webView.evaluateJavascript("window.onModelLoaded('$id', $ok)", null) }
            }
        }

        @JavascriptInterface
        fun openExternalUrl(url: String) {
            runOnUiThread {
                try { startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url))) } catch (_: Exception) {}
            }
        }

        @JavascriptInterface
        fun attachModelFile(targetId: String) {
            pendingAttachTargetId = targetId
            runOnUiThread {
                val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE)
                    type = "*/*"
                }
                attachTargetLauncher.launch(intent)
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
                promptStartTimeMs = System.currentTimeMillis()
                firstTokenTimeMs = 0L
                nativeGenerate(formattedHistory, sysPrompt, rgbBytes, w, h, temp, topP, maxTokens)
                runOnUiThread { webView.evaluateJavascript("window.onVoltDone()", null) }
            }
        }

        @JavascriptInterface
        fun pickCustomModelFile() {
            runOnUiThread { customModelPickerLauncher.launch(arrayOf("*/*")) }
        }

        @JavascriptInterface
        fun isAgentAccessibilityEnabled(): Boolean = VoltAgentService.isEnabled()

        @JavascriptInterface
        fun getScreenUiState(): String = VoltAgentService.dumpScreenTree()

        @JavascriptInterface
        fun captureScreenForAgent(): String = VoltAgentService.captureScreenDataUrl()

        @JavascriptInterface
        fun getWorkspacePath(): String = workspaceDir().absolutePath

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
                val action = obj.optString("action", obj.optString("cmd", "")).lowercase().trim()
                val target = obj.optString("target", obj.optString("path", ""))
                val value = obj.optString("value", obj.optString("content", ""))

                when (action) {
                    "read_screen" -> VoltAgentService.dumpScreenTree()
                    "list_apps" -> getInstalledAppsList()
                    "click_id" -> {
                        val idx = target.filter { it.isDigit() }.toIntOrNull() ?: -1
                        if (!VoltAgentService.isEnabled()) "Нужно включить спец. возможности для Volt AI"
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
                        if (!VoltAgentService.isEnabled()) "Нужно включить спец. возможности для Volt AI"
                        else if (VoltAgentService.clickByText(target)) "Нажато: «$target»"
                        else "Элемент «$target» не найден на экране"
                    }
                    "input_text" -> {
                        val idx = value.filter { it.isDigit() }.toIntOrNull() ?: -1
                        if (!VoltAgentService.isEnabled()) "Нужно включить спец. возможности для Volt AI"
                        else if (VoltAgentService.inputText(target, idx)) "Введён текст: «$target»"
                        else "Поле ввода не найдено"
                    }
                    "scroll" -> {
                        if (!VoltAgentService.isEnabled()) "Нужно включить спец. возможности"
                        else if (VoltAgentService.scrollDown()) "Экран прокручен вниз"
                        else "Не удалось прокрутить экран"
                    }
                    "scroll_up" -> {
                        if (!VoltAgentService.isEnabled()) "Нужно включить спец. возможности"
                        else if (VoltAgentService.scrollUp()) "Экран прокручен вверх"
                        else "Не удалось прокрутить вверх"
                    }
                    "back" -> { VoltAgentService.goBack(); "Выполнено: Назад" }
                    "home" -> { VoltAgentService.goHome(); "Переход на главный экран" }
                    "open_app" -> {
                        val match = findBestMatchingApp(target)
                        if (match != null) {
                            val launch = packageManager.getLaunchIntentForPackage(match.second)
                            if (launch != null) {
                                launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                                startActivity(launch)
                                "Открыто приложение: ${match.first}"
                            } else "Не удалось запустить: ${match.first}"
                        } else "Приложение «$target» не найдено среди установленных"
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
                    "list_files" -> {
                        val dir = resolveWorkspaceFile(target)
                        if (!dir.exists() || !dir.isDirectory) "Папка не найдена: ${dir.absolutePath}"
                        else {
                            val list = dir.listFiles()?.sortedBy { it.name }?.take(60)?.joinToString("\n") { f ->
                                if (f.isDirectory) "[DIR] ${f.name}/" else "${f.name} (${f.length()} B)"
                            }.orEmpty()
                            "Папка ${dir.absolutePath}:\n${list.ifEmpty { "(пусто)" }}"
                        }
                    }
                    "read_file" -> {
                        val f = resolveWorkspaceFile(target)
                        if (!f.exists() || !f.isFile) "Файл не найден: ${f.absolutePath}"
                        else {
                            val content = f.readText()
                            if (content.length > 5000) content.take(5000) + "\n...[обрезано, всего ${content.length} символов]"
                            else content
                        }
                    }
                    "write_file" -> {
                        val f = resolveWorkspaceFile(target)
                        f.parentFile?.mkdirs()
                        f.writeText(value)
                        "Файл сохранён: ${f.absolutePath} (${value.length} симв.)"
                    }
                    "append_file" -> {
                        val f = resolveWorkspaceFile(target)
                        f.parentFile?.mkdirs()
                        f.appendText(value)
                        "Добавлено в файл: ${f.absolutePath}"
                    }
                    "delete_file" -> {
                        val f = resolveWorkspaceFile(target)
                        if (f.exists() && f.delete()) "Удалено: ${f.absolutePath}"
                        else "Не удалось удалить: ${f.absolutePath}"
                    }
                    "run_shell" -> {
                        val cmd = target.ifEmpty { value }
                        if (cmd.isBlank()) return "Пустая shell-команда"
                        val proc = ProcessBuilder("sh", "-c", cmd)
                            .directory(workspaceDir())
                            .redirectErrorStream(true)
                            .start()
                        val finished = proc.waitFor(10, TimeUnit.SECONDS)
                        if (!finished) {
                            proc.destroyForcibly()
                            "Таймаут команды (10 сек): $cmd"
                        } else {
                            val out = proc.inputStream.bufferedReader().readText().trim()
                            val code = proc.exitValue()
                            val clipped = if (out.length > 3500) out.take(3500) + "\n...[обрезано]" else out
                            "Exit code: $code\n${clipped.ifEmpty { "(без вывода)" }}"
                        }
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
}
