package com.voltage.ai

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.graphics.Path
import android.graphics.Rect
import android.os.Bundle
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo

class VoltAgentService : AccessibilityService() {
    companion object {
        var instance: VoltAgentService? = null
        private val cachedNodes = mutableListOf<AccessibilityNodeInfo>()

        fun isEnabled(): Boolean = instance != null

        fun dumpScreenTree(): String {
            val svc = instance ?: return "Экран недоступен (AccessibilityService выключен)"
            val root = svc.rootInActiveWindow ?: return "Активное окно не найдено"
            cachedNodes.clear()
            val sb = StringBuilder()
            val pkg = root.packageName?.toString() ?: "unknown"
            sb.append("Текущее приложение: $pkg\nЭлементы на экране:\n")

            fun traverse(node: AccessibilityNodeInfo?, depth: Int) {
                if (node == null || depth > 14 || cachedNodes.size >= 45) return
                val text = node.text?.toString()?.trim().orEmpty()
                val desc = node.contentDescription?.toString()?.trim().orEmpty()
                val label = when {
                    text.isNotEmpty() && desc.isNotEmpty() && text != desc -> "$text ($desc)"
                    text.isNotEmpty() -> text
                    desc.isNotEmpty() -> desc
                    else -> ""
                }
                val isClickable = node.isClickable
                val isEditable = node.isEditable
                val isScrollable = node.isScrollable

                if (label.isNotEmpty() || isClickable || isEditable) {
                    val idx = cachedNodes.size
                    cachedNodes.add(node)
                    val rect = Rect()
                    node.getBoundsInScreen(rect)
                    val flags = mutableListOf<String>()
                    if (isClickable) flags.add("click")
                    if (isEditable) flags.add("input")
                    if (isScrollable) flags.add("scroll")
                    val shownLabel = if (label.isNotEmpty()) "\"${label.take(45)}\"" else "[без текста]"
                    sb.append("[$idx] $shownLabel (${flags.joinToString(",")}) [${rect.centerX()},${rect.centerY()}]\n")
                }
                for (i in 0 until node.childCount) {
                    traverse(node.getChild(i), depth + 1)
                }
            }

            traverse(root, 0)
            return if (cachedNodes.isEmpty()) "На экране нет распознанных элементов" else sb.toString()
        }

        fun clickById(id: Int): Boolean {
            val node = cachedNodes.getOrNull(id) ?: return false
            var curr: AccessibilityNodeInfo? = node
            while (curr != null) {
                if (curr.isClickable && curr.performAction(AccessibilityNodeInfo.ACTION_CLICK)) return true
                curr = curr.parent
            }
            val rect = Rect()
            node.getBoundsInScreen(rect)
            return tapAt(rect.centerX().toFloat(), rect.centerY().toFloat())
        }

        fun clickByText(targetText: String): Boolean {
            val root = instance?.rootInActiveWindow ?: return false
            val nodes = root.findAccessibilityNodeInfosByText(targetText)
            for (node in nodes) {
                var curr: AccessibilityNodeInfo? = node
                while (curr != null) {
                    if (curr.isClickable && curr.performAction(AccessibilityNodeInfo.ACTION_CLICK)) return true
                    curr = curr.parent
                }
                val rect = Rect()
                node.getBoundsInScreen(rect)
                if (rect.centerX() > 0 && rect.centerY() > 0) {
                    return tapAt(rect.centerX().toFloat(), rect.centerY().toFloat())
                }
            }
            return false
        }

        fun tapAt(x: Float, y: Float): Boolean {
            val svc = instance ?: return false
            val path = Path().apply { moveTo(x, y) }
            val stroke = GestureDescription.StrokeDescription(path, 0, 60)
            val gesture = GestureDescription.Builder().addStroke(stroke).build()
            return svc.dispatchGesture(gesture, null, null)
        }

        fun inputText(text: String, nodeId: Int = -1): Boolean {
            val root = instance?.rootInActiveWindow ?: return false
            val targetNode = if (nodeId >= 0) cachedNodes.getOrNull(nodeId) else root.findFocus(AccessibilityNodeInfo.FOCUS_INPUT)
            val editable = targetNode ?: cachedNodes.firstOrNull { it.isEditable }
            if (editable != null) {
                editable.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
                editable.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                val args = Bundle().apply {
                    putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text)
                }
                return editable.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)
            }
            return false
        }

        fun scrollDown(): Boolean {
            val root = instance?.rootInActiveWindow ?: return false
            return findScrollable(root)?.performAction(AccessibilityNodeInfo.ACTION_SCROLL_FORWARD) ?: false
        }

        fun goBack(): Boolean = instance?.performGlobalAction(GLOBAL_ACTION_BACK) ?: false
        fun goHome(): Boolean = instance?.performGlobalAction(GLOBAL_ACTION_HOME) ?: false

        private fun findScrollable(node: AccessibilityNodeInfo?): AccessibilityNodeInfo? {
            if (node == null) return null
            if (node.isScrollable) return node
            for (i in 0 until node.childCount) {
                val found = findScrollable(node.getChild(i))
                if (found != null) return found
            }
            return null
        }
    }

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {}
    override fun onInterrupt() {}

    override fun onDestroy() {
        instance = null
        super.onDestroy()
    }
}
