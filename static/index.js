document.addEventListener("DOMContentLoaded", async function () {
	/*******************************************************************
	 * 1. 基础变量初始化
	 *******************************************************************/
	const videoStreamEl = document.getElementById("videoStream"); // img 标签
	const startPauseBtn = document.getElementById("startPauseBtn");
	const classificationResultsContainer = document.getElementById(
		"classificationResults"
	);
	const socket = io();
	let isStreaming = false;
	let fpsAnimationId = null; // 用于存储 requestAnimationFrame 的ID
	const videoContainer = document.getElementById("videoContainer");
	let cameraAspectRatio = 4 / 3; // 默认比例

	// 名称映射
	let nameMapping = {};
	let reverseNameMapping = {};

	// 统计数据 (提前定义)
	const stats = {
		characters: new Map(),
		regions: new Map(),
	};

	// 先加载名称映射
	await loadNameMapping();

	// 再初始化统计数据
	loadStats(); // 现在可以安全地使用名称映射了

	/*******************************************************************
	 * 2. 名称转换功能
	 *******************************************************************/
	async function loadNameMapping() {
		try {
			const response = await fetch("/api/name_mapping");
			if (!response.ok) throw new Error("Failed to load name mapping");
			nameMapping = await response.json();

			// 创建反向映射（中文名到英文名）
			reverseNameMapping = {};
			for (const [enName, cnName] of Object.entries(nameMapping)) {
				reverseNameMapping[cnName] = enName;
			}

			console.log("名称映射加载成功");
		} catch (err) {
			console.error("加载名称映射失败:", err);
		}
	}

	// 英文名转中文名
	function enToCn(enName) {
		return nameMapping[enName] || enName;
	}

	// 中文名转英文名
	function cnToEn(cnName) {
		return reverseNameMapping[cnName] || cnName;
	}

	// 获取角色图片路径（使用英文名）
	function getCharacterImagePath(characterName, size = "small") {
		const enName =
			typeof characterName === "string" &&
			reverseNameMapping[characterName]
				? reverseNameMapping[characterName]
				: characterName;
		return `/static/images/${enName}_${size}.png`;
	}

	/*******************************************************************
	 * 3. FPS计算
	 *******************************************************************/
	let fpsDisplay = document.getElementById("fpsDisplay");
	let lastTime = performance.now();
	let frameCount = 0;

	function updateFPS() {
		frameCount++;
		let now = performance.now();
		let diff = now - lastTime;
		if (diff >= 1000) {
			let fps = (frameCount / diff) * 1000;
			fpsDisplay.textContent = "FPS: " + fps.toFixed(1);
			frameCount = 0;
			lastTime = now;
		}
		fpsAnimationId = requestAnimationFrame(updateFPS);
	}

	function stopFPSCount() {
		if (fpsAnimationId) {
			cancelAnimationFrame(fpsAnimationId);
			fpsAnimationId = null;
			fpsDisplay.textContent = "FPS: 0";
		}
	}

	/*******************************************************************
	 * 4. 系统控制
	 *******************************************************************/
	async function getCameraInfo() {
		try {
			const response = await fetch("/api/camera_info");
			if (!response.ok) throw new Error("Failed to get camera info");
			const data = await response.json();
			cameraAspectRatio = data.width / data.height;
			updateVideoContainerSize();
		} catch (err) {
			console.error("Error getting camera info:", err);
		}
	}

	function updateVideoContainerSize() {
		const containerWidth = videoContainer.offsetWidth;
		const containerHeight = containerWidth / cameraAspectRatio;
		videoContainer.style.height = `${containerHeight}px`;
	}

	async function startSystem() {
		try {
			// 获取摄像头信息
			await getCameraInfo();

			// 加载名称映射
			await loadNameMapping();

			// 1. 启动后端推理
			// console.log("启动推理");
			const response = await fetch("/api/startProcessing", {
				method: "POST",
			});
			if (!response.ok) throw new Error("启动推理失败");

			// 2. 启动视频流
			// console.log("启动视频流");
			videoStreamEl.src = "/video_feed";
			// const sourceObj = document.createElement("source");
			// sourceObj.setAttribute("src", "/video_feed");
			// videoStreamEl.appendChild(sourceObj);
			// await videoStreamEl.play();  // 等待视频开始播放

			// 3. 启动FPS计算
			// console.log("启动FPS计算");
			updateFPS();

			// 4. 更新UI状态
			// console.log("更新UI状态");
			isStreaming = true;
			startPauseBtn.textContent = "暂停";
			startPauseBtn.classList.remove("bg-blue-500", "hover:bg-blue-600");
			startPauseBtn.classList.add("bg-red-500", "hover:bg-red-600");

			// 5. 重置模型FPS显示
			// console.log("重置模型FPS显示");
			document.getElementById("modelFpsDisplay").textContent =
				"Model FPS: 0";
			lastInferenceTime = Date.now();
			inferenceFrameCount = 0;
		} catch (err) {
			console.error("系统启动失败:", err);
			await stopSystem(); // 发生错误时确保系统完全停止
		}
	}

	async function stopSystem() {
		try {
			// 1. 停止后端推理
			await fetch("/api/stopProcessing", { method: "POST" });

			// 2. 停止视频流
			videoStreamEl.src = "";
			// videoStreamEl.load();  // 确保视频完全停止

			// 3. 停止FPS计算
			stopFPSCount();

			// 4. 清空分类结果
			classificationResultsContainer.innerHTML = "";
			document.getElementById("modelFpsDisplay").textContent =
				"Model FPS: 0";

			// 5. 更新UI状态
			isStreaming = false;
			startPauseBtn.textContent = "启动";
			startPauseBtn.classList.remove("bg-red-500", "hover:bg-red-600");
			startPauseBtn.classList.add("bg-blue-500", "hover:bg-blue-600");
		} catch (err) {
			console.error("系统停止失败:", err);
		}
	}

	/*******************************************************************
	 * 4. 启动/暂停按钮控制
	 *******************************************************************/
	startPauseBtn.addEventListener("click", async () => {
		if (!isStreaming) {
			await startSystem();
		} else {
			await stopSystem();
		}
	});

	// 页面关闭时确保系统停止
	window.addEventListener("beforeunload", async () => {
		if (isStreaming) {
			await stopSystem();
		}
	});

	// 添加窗口大小变化的监听
	window.addEventListener("resize", updateVideoContainerSize);

	/*******************************************************************
	 * 5. 推理结果处理
	 *******************************************************************/
	let lastInferenceTime = Date.now();
	let inferenceFrameCount = 0;

	socket.on("inference_result", (data) => {
		if (isStreaming) {
			// 只在系统运行时更新结果
			updateClassificationResults(data.top5);
			calculateModelFPS();
		}
	});

	// TODO: 优化更新机制，只更新textContent和src
	function updateClassificationResults(top5Array) {
		classificationResultsContainer.innerHTML = "";
		top5Array.forEach((item, index) => {
			const row = document.createElement("div");
			if (index === 0) {
				row.className = "mb-4";

				const wordRow = document.createElement("div");
				wordRow.className = "flex items-center justify-between mb-2";

				const leftSection = document.createElement("div");
				leftSection.className = "flex items-center space-x-2";

				const rank = document.createElement("span");
				rank.className = "font-bold w-6";
				rank.textContent = `#${index + 1}`;
				leftSection.appendChild(rank);

				const name = document.createElement("span");
				name.className = "top1-name";
				name.textContent = item.classNameCN || enToCn(item.className);
				leftSection.appendChild(name);

				wordRow.appendChild(leftSection);

				const rightSection = document.createElement("span");
				rightSection.className = "text-gray-600";
				rightSection.textContent = `${(item.probability * 100).toFixed(
					1
				)}%`;
				wordRow.appendChild(rightSection);

				row.appendChild(wordRow);

				const image = document.createElement("img");
				image.src = getCharacterImagePath(item.className, "large");
				image.className = "w-full object-contain rounded-lg shadow-sm";
				image.style = "max-height: 200px";
				row.appendChild(image);
			} else {
				row.className =
					"flex items-center justify-between p-2 bg-gray-50 rounded";

				const leftSection = document.createElement("div");
				leftSection.className = "flex items-center space-x-2";

				const rank = document.createElement("span");
				rank.className = "font-bold w-6";
				rank.textContent = `#${index + 1}`;
				leftSection.appendChild(rank);

				const name = document.createElement("span");
				name.textContent = item.classNameCN;
				leftSection.appendChild(name);

				row.appendChild(leftSection);

				const rightSection = document.createElement("div");
				rightSection.className = "flex items-center space-x-2";

				const probability = document.createElement("span");
				probability.className = "text-gray-600";
				probability.textContent = `${(item.probability * 100).toFixed(
					1
				)}%`;
				rightSection.appendChild(probability);

				const image = document.createElement("img");
				image.src = getCharacterImagePath(item.className, "small");
				image.className = "w-8 h-8 object-cover rounded";
				rightSection.appendChild(image);

				row.appendChild(rightSection);
			}
			classificationResultsContainer.appendChild(row);
		});

		// // 创建 top1 结果的特殊显示
		// const top1 = top5Array[0];
		// const top1El = document.createElement("div");
		// top1El.className = "mb-4";
		// top1El.innerHTML = `
		// 	<div class="flex items-center justify-between mb-2">
		// 		<div class="flex items-center space-x-2">
		// 			<span class="font-bold w-6">#1</span>
		// 			<span class="top1-name">${top1.className}</span>
		// 		</div>
		// 		<span class="text-gray-600">${(top1.probability * 100).toFixed(1)}%</span>
		// 	</div>
		// 	<img src="/static/images/${top1.className}_large.png"
		// 		 class="w-full object-contain rounded-lg shadow-sm"
		// 		 style="max-height: 200px" />
		// `;
		// classificationResultsContainer.appendChild(top1El);
		// 创建其他结果的显示
		// const othersEl = document.createElement("div");
		// othersEl.className = "space-y-2";
		// // 添加信息
		// top5Array.slice(1).forEach((item, index) => {
		// 	const row = document.createElement("div");
		// 	row.className =
		// 		"flex items-center justify-between p-2 bg-gray-50 rounded";
		// 	row.innerHTML = `
		// 		<div class="flex items-center space-x-2">
		// 			<span class="font-bold w-6">#${index + 2}</span>
		// 			<span>${item.className}</span>
		// 		</div>
		// 		<div class="flex items-center space-x-2">
		// 			<span class="text-gray-600">${(item.probability * 100).toFixed(1)}%</span>
		// 			<img src="/static/images/${
		// 				item.className
		// 			}_small.png" class="w-8 h-8 object-cover rounded" />
		// 		</div>
		// 	`;
		// 	othersEl.appendChild(row);
		// });
		// classificationResultsContainer.appendChild(othersEl);
	}

	function calculateModelFPS() {
		inferenceFrameCount++;
		const now = Date.now();
		const diff = now - lastInferenceTime;
		if (diff >= 1000) {
			const modelFps = (inferenceFrameCount / diff) * 1000;
			document.getElementById("modelFpsDisplay").textContent =
				"Model FPS: " + modelFps.toFixed(1);
			inferenceFrameCount = 0;
			lastInferenceTime = now;
		}
	}

	/*******************************************************************
	 * 6. 画廊功能
	 *******************************************************************/
	const galleryEl = document.getElementById("gallery");

	// 添加图片到画廊的函数
	function addImageToGallery(canvas, top1Name) {
		const wrapper = document.createElement("div");
		wrapper.className = "relative group flex-shrink-0";

		// 创建缩略图
		const thumbnail = document.createElement("img");
		thumbnail.src = canvas.toDataURL("image/png");
		thumbnail.className =
			"h-40 object-contain border hover:border-blue-500 cursor-pointer rounded";

		// 点击查看大图
		thumbnail.onclick = () => {
			const modal = document.createElement("div");
			modal.className =
				"fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50";

			const modalImg = document.createElement("img");
			modalImg.src = canvas.toDataURL("image/png");
			modalImg.className = "max-h-[90vh] max-w-[90vw] object-contain";

			modal.appendChild(modalImg);
			modal.onclick = () => modal.remove();
			document.body.appendChild(modal);
		};

		wrapper.appendChild(thumbnail);
		galleryEl.insertBefore(wrapper, galleryEl.firstChild);

		// 自动滚动到最新的图片
		const galleryContainer = galleryEl.parentElement;
		galleryContainer.scrollTo({
			left: 0,
			behavior: "smooth",
		});

		// 更新统计
		if (top1Name) {
			updateStats(top1Name);
		}
	}

	/*******************************************************************
	 * 7. 截图功能
	 *******************************************************************/
	const snapshotBtn = document.getElementById("snapshotBtn");

	// 创建快照
	function createSnapshotImage(videoEl, classificationResults) {
		const canvas = document.createElement("canvas");
		const ctx = canvas.getContext("2d");

		// 视频尺寸
		const videoWidth = 720; // 固定视频宽度
		const videoHeight = 480; // 固定视频高度
		const resultWidth = 360; // 分类结果区域宽度

		// 计算整体尺寸
		const totalWidth = videoWidth + resultWidth;
		const targetHeight = Math.round(totalWidth * 2 / 3); // 3:2比例的目标高度
		const padding = Math.round((targetHeight - videoHeight) / 2); // 上下白边高度

		// 设置画布尺寸
		canvas.width = totalWidth;
		canvas.height = targetHeight;

		// 绘制白色背景
		ctx.fillStyle = "white";
		ctx.fillRect(0, 0, canvas.width, canvas.height);

		// 绘制视频帧（居中）
		ctx.drawImage(videoEl, 0, padding, videoWidth, videoHeight);

		// 绘制分割线
		ctx.strokeStyle = "#e5e7eb";
		ctx.lineWidth = 2;
		ctx.beginPath();
		ctx.moveTo(videoWidth, 0);
		ctx.lineTo(videoWidth, targetHeight);
		ctx.stroke();

		// 如果有分类结果，绘制第一个结果
		if (classificationResults.children.length > 0) {
			const topResult = classificationResults.children[0];
			const className = topResult.querySelector(".top1-name").textContent;
			const probability = topResult.querySelector(".text-gray-600").textContent;

			// 设置文字样式
			ctx.fillStyle = "#111827";
			ctx.font = "bold 24px Arial";
			ctx.fillText("#1 " + className, videoWidth + 20, padding + 40);

			ctx.font = "20px Arial";
			ctx.fillText(probability, videoWidth + 20, padding + 80);

			// 绘制角色图片
			const img = topResult.querySelector("img");
			const imgWidth = 250;
			const imgHeight = imgWidth * (img.naturalHeight / img.naturalWidth);
			ctx.drawImage(img, videoWidth + 40, padding + 100, imgWidth, imgHeight);
		}

		return canvas;
	}

	// 添加截图按钮事件监听
	snapshotBtn.addEventListener("click", async () => {
		if (!isStreaming) return; // 如果没有在运行，直接返回

		// 创建快照
		const canvas = createSnapshotImage(
			videoStreamEl,
			classificationResultsContainer
		);

		// 获取当前的top1分类结果
		const top1NameEl = document.querySelector(
			"#classificationResults .top1-name"
		);
		if (top1NameEl) {
			const top1Name = top1NameEl.textContent.trim();
			addImageToGallery(canvas, top1Name);
			
			// 保存图片到服务器
			try {
				const response = await fetch('/api/save_image', {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json',
					},
					body: JSON.stringify({
						image: canvas.toDataURL('image/png')
					})
				});
				
				const result = await response.json();
				if (result.status === 'success') {
					console.log('图片已保存:', result.filename);
				}
			} catch (error) {
				console.error('保存图片失败:', error);
			}
		}
	});

	// 监听图片保存事件
	socket.on('image_saved', (data) => {
		console.log('新图片已保存:', data.filename);
		// 这里可以添加通知UI的代码
		const notification = document.createElement('div');
		notification.className = 'fixed bottom-4 right-4 bg-green-500 text-white px-4 py-2 rounded-lg shadow-lg';
		notification.textContent = `图片已保存: ${data.filename}`;
		document.body.appendChild(notification);
		
		// 3秒后移除通知
		setTimeout(() => {
			notification.remove();
		}, 3000);
	});

	/*******************************************************************
	 * 统计数据管理
	 *******************************************************************/
	// const stats = { // <--- Remove this later definition
	// 	characters: new Map(), // 角色统计
	// 	regions: new Map(), // 地域统计
	// };

	// 角色与地域的映射关系
	const characterRegions = {
		// 示例映射，需要根据实际情况补充
		aki_minoriko: "妖怪之山（山麓）",
		aki_shizuha: "妖怪之山（山麓）",
		alice_margatroid: "魔法之森",
		asakura_rikako: "人间之里",
		chen: "三途之川~冥界",
		cirno: "雾之湖~红魔馆",
		clownpiece: "梦之世界~月",
		daiyousei: "雾之湖~红魔馆",
		doremy_sweet: "梦之世界~月",
		ebisu_eika: "三途之川~冥界",
		elis: "妖怪之山（山顶）",
		elly: "迷途竹林",
		eternity_larva: "迷途竹林",
		flandre_scarlet: "雾之湖~红魔馆",
		fujiwara_no_mokou: "迷途竹林",
		futatsuiwa_mamizou: "命莲寺",
		gengetsu: "迷途竹林",
		hakurei_reimu: "博丽神社",
		haniyasushin_keiki: "三途之川~冥界",
		hata_no_kokoro: "博丽神社",
		hecatia_lapislazuli: "梦之世界~月",
		hieda_no_akyuu: "人间之里",
		hijiri_byakuren: "命莲寺",
		himekaidou_hatate: "妖怪之山（山顶）",
		himemushi_momoyo: "妖怪之山（山顶）",
		hinanawi_tenshi: "妖怪之山（山顶）",
		hong_meiling: "雾之湖~红魔馆",
		horikawa_raiko: "人间之里",
		hoshiguma_yuugi: "地底",
		houjuu_nue: "命莲寺",
		houraisan_kaguya: "迷途竹林",
		ibaraki_kasen: "妖怪之山（山麓）",
		ibuki_suika: "博丽神社",
		iizunamaru_megumu: "妖怪之山（山顶）",
		imaizumi_kagerou: "迷途竹林",
		inaba_tewi: "迷途竹林",
		inubashiri_momiji: "妖怪之山（山顶）",
		izayoi_sakuya: "雾之湖~红魔馆",
		joutougu_mayumi: "三途之川~冥界",
		junko: "梦之世界~月",
		kaenbyou_rin: "地底",
		kagiyama_hina: "妖怪之山（山麓）",
		kaku_seiga: "命莲寺",
		kamishirasawa_keine: "人间之里",
		kana_anaberal: "博丽神社",
		kasodani_kyouko: "命莲寺",
		kawashiro_nitori: "妖怪之山（山麓）",
		kazami_yuuka: "迷途竹林",
		kicchou_yachie: "三途之川~冥界",
		kijin_seija: "妖怪之山（山麓）",
		kirisame_marisa: "魔法之森",
		kishin_sagume: "梦之世界~月",
		kisume: "地底",
		kitashirakawa_chiyuri: "博丽神社",
		koakuma: "雾之湖~红魔馆",
		kochiya_sanae: "妖怪之山（山顶）",
		komakusa_sannyo: "妖怪之山（山顶）",
		komano_aunn: "博丽神社",
		komeiji_koishi: "地底",
		komeiji_satori: "地底",
		konngara: "三途之川~冥界",
		konpaku_youmu: "三途之川~冥界",
		kotohime: "人间之里",
		kudamaki_tsukasa: "妖怪之山（山顶）",
		kumoi_ichirin: "命莲寺",
		kurokoma_saki: "三途之川~冥界",
		kurumi: "迷途竹林",
		letty_whiterock: "雾之湖~红魔馆",
		lily_white: "妖怪之山（山麓）",
		luize: "人间之里",
		lunasa_prismriver: "雾之湖~红魔馆",
		luna_child: "博丽神社",
		lyrica_prismriver: "雾之湖~红魔馆",
		mai: "人间之里",
		maribel_hearn: "人间之里",
		matara_okina: "三途之川~冥界",
		medicine_melancholy: "迷途竹林",
		meira: "人间之里",
		merlin_prismriver: "雾之湖~红魔馆",
		mima: "博丽神社",
		miyako_yoshika: "命莲寺",
		mononobe_no_futo: "命莲寺",
		moriya_suwako: "妖怪之山（山顶）",
		motoori_kosuzu: "人间之里",
		mugetsu: "迷途竹林",
		murasa_minamitsu: "命莲寺",
		nagae_iku: "妖怪之山（山顶）",
		nazrin: "命莲寺",
		nishida_satono: "三途之川~冥界",
		niwatari_kutaka: "妖怪之山（山顶）",
		okazaki_yumemi: "博丽神社",
		okunoda_miyoi: "人间之里",
		onozuka_komachi: "三途之川~冥界",
		orange: "妖怪之山（山麓）",
		patchouli_knowledge: "雾之湖~红魔馆",
		reisen: "梦之世界~月",
		reisen_udongein_inaba: "迷途竹林",
		reiuji_utsuho: "地底",
		remilia_scarlet: "雾之湖~红魔馆",
		rika: "人间之里",
		ringo: "梦之世界~月",
		rumia: "魔法之森",
		ruukoto: "博丽神社",
		saigyouji_yuyuko: "三途之川~冥界",
		sakata_nemuno: "妖怪之山（山麓）",
		sara: "人间之里",
		sariel: "妖怪之山（山顶）",
		satsuki_rin: "人间之里",
		seiran: "梦之世界~月",
		sekibanki: "人间之里",
		shameimaru_aya: "妖怪之山（山顶）",
		shiki_eiki: "三途之川~冥界",
		shinki: "魔法之森",
		soga_no_tojiko: "命莲寺",
		star_sapphire: "博丽神社",
		sunny_milk: "博丽神社",
		tamatsukuri_misumaru: "妖怪之山（山顶）",
		tatara_kogasa: "命莲寺",
		teireida_mai: "三途之川~冥界",
		tenkyuu_chimata: "妖怪之山（山顶）",
		tokiko: "魔法之森",
		toutetsu_yuuma: "三途之川~冥界",
		toyosatomimi_no_miko: "命莲寺",
		tsukumo_benben: "人间之里",
		tsukumo_yatsuhashi: "人间之里",
		usami_sumireko: "人间之里",
		ushizaki_urumi: "三途之川~冥界",
		wakasagihime: "雾之湖~红魔馆",
		watatsuki_no_toyohime: "梦之世界~月",
		watatsuki_no_yorihime: "梦之世界~月",
		wriggle_nightbug: "魔法之森",
		yagokoro_eirin: "迷途竹林",
		yakumo_ran: "三途之川~冥界",
		yakumo_yukari: "三途之川~冥界",
		yamashiro_takane: "妖怪之山（山麓）",
		yasaka_kanako: "妖怪之山（山顶）",
		yatadera_narumi: "魔法之森",
		yorigami_shion: "妖怪之山（山顶）",
		yuki: "人间之里",
		yumeko: "魔法之森",
		mystia_lorelei: "迷途竹林",
		kurodani_yamame: "地底",
		mitsugashira_enoko: "魔法之森",
		mizuhashi_parsee: "地底",
		morichika_rinnosuke: "魔法之森",
		nippaku_zanmu: "三途之川~冥界",
		son_biten: "妖怪之山（山麓）",
		sukuna_shinmyoumaru: "博丽神社",
		tenkajin_chiyari: "地底",
		toramaru_shou: "命莲寺",
		usami_renko: "人间之里",
		yomotsu_hisami: "三途之川~冥界",
		yorigami_jyoon: "命莲寺",
	};

	// 地域与地图区域的映射关系
	const regionMapAreas = {
		博丽神社: [
			{ row: 28, startCol: 28, endCol: 31 },
			{ row: 29, startCol: 28, endCol: 31 },
			{ row: 30, startCol: 28, endCol: 31 },
		],
		命莲寺: [
			{ row: 27, startCol: 11, endCol: 14 },
			{ row: 28, startCol: 11, endCol: 14 },
			{ row: 29, startCol: 11, endCol: 14 },
			{ row: 30, startCol: 11, endCol: 14 },
		],
		人间之里: [
			{ row: 25, startCol: 16, endCol: 22 },
			{ row: 26, startCol: 16, endCol: 22 },
			{ row: 27, startCol: 16, endCol: 22 },
			{ row: 28, startCol: 16, endCol: 22 },
		],
		"雾之湖~红魔馆": [
			{ row: 15, startCol: 30, endCol: 34 },
			{ row: 16, startCol: 29, endCol: 33 },
			{ row: 17, startCol: 27, endCol: 31 },
			{ row: 18, startCol: 27, endCol: 31 },
			{ row: 19, startCol: 27, endCol: 31 },
			{ row: 20, startCol: 27, endCol: 31 },
			{ row: 21, startCol: 27, endCol: 31 },
		],
		迷途竹林: [
			{ row: 22, startCol: 4, endCol: 8 },
			{ row: 23, startCol: 4, endCol: 8 },
			{ row: 24, startCol: 4, endCol: 8 },
			{ row: 25, startCol: 4, endCol: 8 },
			{ row: 26, startCol: 4, endCol: 8 },
		],
		魔法之森: [
			{ row: 15, startCol: 13, endCol: 20 },
			{ row: 16, startCol: 13, endCol: 20 },
			{ row: 17, startCol: 11, endCol: 18 },
			{ row: 18, startCol: 11, endCol: 18 },
			{ row: 19, startCol: 11, endCol: 18 },
			{ row: 20, startCol: 11, endCol: 18 },
		],
		"三途之川~冥界": [
			{ row: 2, startCol: 2, endCol: 14 },
			{ row: 3, startCol: 2, endCol: 14 },
			{ row: 4, startCol: 2, endCol: 14 },
			{ row: 5, startCol: 2, endCol: 15 },
			{ row: 6, startCol: 2, endCol: 16 },
			{ row: 7, startCol: 2, endCol: 19 },
		],
		"妖怪之山（山麓）": [
			{ row: 9, startCol: 32, endCol: 36 },
			{ row: 10, startCol: 29, endCol: 33 },
			{ row: 11, startCol: 23, endCol: 27 },
			{ row: 12, startCol: 21, endCol: 25 },
			{ row: 13, startCol: 21, endCol: 25 },
		],
		"妖怪之山（山顶）": [
			{ row: 4, startCol: 20, endCol: 24 },
			{ row: 5, startCol: 21, endCol: 26 },
			{ row: 6, startCol: 23, endCol: 29 },
			{ row: 7, startCol: 25, endCol: 32 },
		],
		地底: [
			{ row: 8, startCol: 28, endCol: 31 },
			{ row: 9, startCol: 27, endCol: 30 },
			{ row: 10, startCol: 22, endCol: 27 },
			{ row: 11, startCol: 18, endCol: 21 },
		],
		"梦之世界~月": [
			{ row: 2, startCol: 16, endCol: 21 },
			{ row: 3, startCol: 17, endCol: 19 },
		],
	};

	// 更新统计数据
	async function updateStats(character) {
		// 更新角色统计
		const charCount = stats.characters.get(character) || 0;
		stats.characters.set(character, charCount + 1);

		// 更新地域统计
		const region = characterRegions[cnToEn(character)];
		if (region) {
			const regionCount = stats.regions.get(region) || 0;
			stats.regions.set(region, regionCount + 1);
		}

		// 向后端发送统计数据
		try {
			await fetch('/api/update_stats', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify({
					character: character,
					region: region
				})
			});
		} catch (error) {
			console.error('更新统计数据失败:', error);
		}

		// 更新显示
		updateStatsDisplay();
		updateMapHighlight(region);
	}

	// 加载统计数据
	async function loadStats() {
		try {
			const response = await fetch('/api/get_stats');
			const data = await response.json();
			
			// 更新角色统计
			stats.characters.clear();
			data.characters.forEach(stat => {
				stats.characters.set(stat.name, stat.count);
			});
			
			// 更新地域统计
			stats.regions.clear();
			data.regions.forEach(stat => {
				stats.regions.set(stat.name, stat.count);
			});
			
			// 更新显示
			updateStatsDisplay();
		} catch (error) {
			console.error('加载统计数据失败:', error);
		}
	}

	// 更新统计显示
	function updateStatsDisplay() {
		// 角色统计显示
		const charStats = Array.from(stats.characters.entries())
			.sort((a, b) => b[1] - a[1])
			.slice(0, 5);

		const characterStatsEl = document.getElementById("characterStats");
		characterStatsEl.innerHTML = charStats
			.map(
				([char, count], index) => `
			<div class="flex items-center justify-between p-2 bg-gray-50 rounded">
				<div class="flex items-center space-x-2">
					<span class="font-bold">#${index + 1}</span>
					<span>${char}</span>
					<span class="text-gray-600">[${count}]</span>
				</div>
				<img src="${getCharacterImagePath(
					cnToEn(char),
					"small"
				)}" class="w-8 h-8 object-cover rounded" />
			</div>
		`
			)
			.join("");

		// 地域统计显示
		const regionStats = Array.from(stats.regions.entries())
			.sort((a, b) => b[1] - a[1])
			.slice(0, 5);

		const regionStatsEl = document.getElementById("regionStats");
		regionStatsEl.innerHTML = regionStats
			.map(
				([region, count], index) => `
			<div class="flex items-center justify-between p-2 bg-gray-50 rounded">
				<div class="flex items-center space-x-2">
					<span class="font-bold">#${index + 1}</span>
					<span>${region}</span>
					<span class="text-gray-600">[${count}]</span>
				</div>
				<div class="w-8 h-8 rounded" style="background-color: ${getRegionColor(
					region
				)}"></div>
			</div>
		`
			)
			.join("");
	}

	// 更新地图高亮
	function updateMapHighlight(region) {
		const areas = regionMapAreas[region] || [];
		// const regionColor = getRegionColor(region);
		
		// 清除所有高亮
		document.querySelectorAll(".map-text").forEach((el) => {
			el.classList.remove("highlight");
			// el.style.backgroundColor = "transparent";
		});

		// 添加新的高亮
		areas.forEach((area) => {
			const rowElements = document.querySelectorAll(
				`.map-text[data-row="${area.row}"]`
			);
			rowElements.forEach((el) => {
				const col = parseInt(el.getAttribute("data-col"));
				if (col >= area.startCol && col <= area.endCol) {
					el.classList.add("highlight");
					// el.style.backgroundColor = regionColor;
				}
			});
		});
	}

	/*******************************************************************
	 * 11. 文字地图初始化
	 *******************************************************************/
	function initializeTextMap() {
		// 从map.txt读取地图数据
		fetch("/static/map.txt")
			.then((response) => response.text())
			.then((text) => {
				const lines = text.split("\n").filter((line) => line.trim());
				const mapEl = document.getElementById("textMap");

				mapEl.innerHTML = lines
					.map((line, rowIndex) => {
						const chars = Array.from(line);
						return `<div class="flex justify-center">
						${chars
							.map((char, colIndex) => {
								const isHalfWidth = char.match(
									/[\x01-\x7E\uFF65-\uFF9F]/
								); // 判断是否为半角字符
								return `<span class="map-text ${
									isHalfWidth ? "half-width" : ""
								}"
								  data-row="${rowIndex + 1}"
								  data-col="${colIndex + 1}"
								  style="color: ${getCharacterColor(char)}">
								${char}
							</span>`;
							})
							.join("")}
					</div>`;
					})
					.join("");
			})
			.catch((err) => console.error("Failed to load map:", err));
	}

	initializeTextMap();

	// TODO:获取地域颜色的函数
	function getRegionColor(region) {
		const colorMap = {
			博丽神社: "#ff9999", // 神社区域用红色
			命莲寺: "#ffcc66",
			人间之里: "#BE5725",
			"雾之湖~红魔馆": "#99ffff",
			迷途竹林: "#66ff99",
			魔法之森: "#00cc00",
			"三途之川~冥界": "#ccffff",
			"妖怪之山（山麓）": "#66ff00",
			"妖怪之山（山顶）": "#cccc33",
			地底: "#666633",
			"梦之世界~月": "#ffffcc",
		};
		return colorMap[region] || "#cccccc"; // 默认灰色
	}

	// 获取字符颜色的函数
	function getCharacterColor(char) {
		const colorMap = {
			树: "#2d5a27", // 深绿色
			林: "#2d5a27",
			森: "#1a4314", // 更深的绿色
			湖: "#1e88e5", // 蓝色
			神: "#d32f2f", // 红色
			社: "#d32f2f",
			魔: "#9c27b0", // 紫色
			法: "#9c27b0",
			妖: "#ff9800", // 橙色
			精: "#ff9800",
			合: "#433322", // 棕色
			言: "#433322",
			ρ: "#c8b865",
			1: "#5c5c9f",
			2: "#5c5c9f",
			3: "#5c5c9f",
			4: "#5c5c9f",
			5: "#5c5c9f",
			6: "#5c5c9f",
			7: "#5c5c9f",
			8: "#5c5c9f",
			9: "#5c5c9f",
			0: "#5c5c9f",
			Ω: "#5c5c9f",
			// "┃": "47291f",
			// "━": "47291f",
			三: "#4386c8",
			"(": "#4386c8",
			ﾐ: "#4386c8",
			"|": "#4386c8",
			ミ: "#4b1900",
			川: "#4b1900",
			火: "#4b1900",
			シ: "#4b1900",
			州: "#4b1900",
			門: "#640000",
			旦: "#88443a",
			"∬": "#640164",
			"＝": "#38221a",

			// ... 其他字符颜色映射
		};
		return colorMap[char] || "#000000"; // 默认黑色
	}

	// // 添加画廊滚动功能
	// document.addEventListener("DOMContentLoaded", function () {
	// 	const galleryContainer = document.querySelector(".overflow-x-auto");
	// 	let isScrolling = false;
	// 	let startX;
	// 	let scrollLeft;

	// 	// 鼠标按下事件
	// 	galleryContainer.addEventListener("mousedown", (e) => {
	// 		isScrolling = true;
	// 		startX = e.pageX - galleryContainer.offsetLeft;
	// 		scrollLeft = galleryContainer.scrollLeft;
	// 		galleryContainer.style.cursor = "grabbing";
	// 	});

	// 	// 鼠标移动事件
	// 	galleryContainer.addEventListener("mousemove", (e) => {
	// 		if (!isScrolling) return;
	// 		e.preventDefault();
	// 		const x = e.pageX - galleryContainer.offsetLeft;
	// 		const walk = (x - startX) * 2; // 滚动速度
	// 		galleryContainer.scrollLeft = scrollLeft - walk;
	// 	});

	// 	// 鼠标释放事件
	// 	galleryContainer.addEventListener("mouseup", () => {
	// 		isScrolling = false;
	// 		galleryContainer.style.cursor = "grab";
	// 	});

	// 	galleryContainer.addEventListener("mouseleave", () => {
	// 		isScrolling = false;
	// 		galleryContainer.style.cursor = "grab";
	// 	});

	// 	// 初始化时设置鼠标样式
	// 	galleryContainer.style.cursor = "grab";
	// });
});
