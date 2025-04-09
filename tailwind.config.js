/** @type {import('tailwindcss').Config} */
module.exports = {
	content: ["./templates/**/*.html", "./static/**/*.js"],
	theme: {
		extend: {
			colors: {
				"region-red": "#ff9999", // 博丽神社
				"region-orange": "#ffcc66", // 命莲寺
				"region-brown": "#BE5725", // 人间之里
				"region-blue": "#99ffff", // 雾之湖~红魔馆
				"region-green": "#66ff99", // 迷途竹林
				"region-darkgreen": "#00cc00", // 魔法之森
				"region-lightblue": "#ccffff", // 三途之川~冥界
				"region-lime": "#66ff00", // 妖怪之山（山麓）
				"region-yellow": "#cccc33", // 妖怪之山（山顶）
				"region-darkbrown": "#666633", // 地底
				"region-lightyellow": "#ffffcc", // 梦之世界~月
				"mapbg": "#B7BBA9",
				"rightbg": "#FEF1E1",
				"resultsbg": "#B6BBBF",
        "topbg": "#DADBD4",
        "allbg": "#FFF9E3",
        "countbg": "#F2D1BA",
        "gallerybg":"#BBBEC5",
			},
		},
	},
	plugins: [],
};
