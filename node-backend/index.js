// node-backend/index.js
const express = require("express");
const multer = require("multer");
const cors = require("cors");
const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs");
const path = require("path");

const app = express();
const upload = multer({ dest: "uploads/" });

app.use(cors());
app.use(express.json());

// API: React가 호출할 업로드 엔드포인트
app.post("/api/upload", upload.single("file"), async (req, res) => {
  try {
    const filePath = path.join(__dirname, req.file.path);

    const formData = new FormData();
    formData.append("file", fs.createReadStream(filePath));

    const flaskResponse = await axios.post("http://127.0.0.1:5000/predict", formData, {
      headers: formData.getHeaders(),
    });

    res.json(flaskResponse.data);
  } catch (err) {
    console.error("🔥 Flask 연결 실패:", err.message);
    res.status(500).json({ error: "Flask 서버와 통신 오류" });
  }
});

// 기본 루트
app.get("/", (req, res) => {
  res.send("🌐 Node.js 중계 API 서버 작동 중");
});

// 서버 실행
app.listen(3001, () => {
  console.log("✅ Node.js 중계 API 서버 실행 중 → http://localhost:3001");
});
