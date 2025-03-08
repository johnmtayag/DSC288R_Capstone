export const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "DSC288R_Capstone/_app",
	assets: new Set(["favicon.png","gradcam/gradcam_metadata.json","gradcam/patientID/resnet50/raw/224/gradcam_condition_0_Enlarged Cardiomediastinum.png","gradcam/patientID/resnet50/raw/224/gradcam_condition_10_Pleural Other.png","gradcam/patientID/resnet50/raw/224/gradcam_condition_11_Fracture.png","gradcam/patientID/resnet50/raw/224/gradcam_condition_12_Support Devices.png","gradcam/patientID/resnet50/raw/224/gradcam_condition_13_No Finding.png","gradcam/patientID/resnet50/raw/224/gradcam_condition_1_Cardiomegaly.png","gradcam/patientID/resnet50/raw/224/gradcam_condition_2_Lung Opacity.png","gradcam/patientID/resnet50/raw/224/gradcam_condition_3_Lung Lesion.png","gradcam/patientID/resnet50/raw/224/gradcam_condition_4_Edema.png","gradcam/patientID/resnet50/raw/224/gradcam_condition_5_Consolidation.png","gradcam/patientID/resnet50/raw/224/gradcam_condition_6_Pneumonia.png","gradcam/patientID/resnet50/raw/224/gradcam_condition_7_Atelectasis.png","gradcam/patientID/resnet50/raw/224/gradcam_condition_8_Pneumothorax.png","gradcam/patientID/resnet50/raw/224/gradcam_condition_9_Pleural Effusion.png","gradcam/patientID/resnet50/raw/224/original_xray.png","summary.json","X-rays_chest_icon.svg"]),
	mimeTypes: {".png":"image/png",".json":"application/json",".svg":"image/svg+xml"},
	_: {
		client: {start:"_app/immutable/entry/start.Bp8uIuv0.js",app:"_app/immutable/entry/app.DO_QAK5x.js",imports:["_app/immutable/entry/start.Bp8uIuv0.js","_app/immutable/chunks/C3Dpbyvy.js","_app/immutable/chunks/BA-I5zKC.js","_app/immutable/chunks/_48DR0pv.js","_app/immutable/chunks/DIeogL5L.js","_app/immutable/chunks/CkqiD5Nf.js","_app/immutable/entry/app.DO_QAK5x.js","_app/immutable/chunks/_48DR0pv.js","_app/immutable/chunks/DIeogL5L.js","_app/immutable/chunks/DcVUTUmq.js","_app/immutable/chunks/CIy1_CLV.js","_app/immutable/chunks/Bg9kRutz.js","_app/immutable/chunks/CZkquXyP.js","_app/immutable/chunks/CIXv7Mok.js","_app/immutable/chunks/Te2Uy4QI.js","_app/immutable/chunks/D052_caE.js","_app/immutable/chunks/BA-I5zKC.js"],stylesheets:[],fonts:[],uses_env_dynamic_public:false},
		nodes: [
			__memo(() => import('./nodes/0.js')),
			__memo(() => import('./nodes/1.js')),
			__memo(() => import('./nodes/2.js')),
			__memo(() => import('./nodes/3.js')),
			__memo(() => import('./nodes/4.js')),
			__memo(() => import('./nodes/5.js')),
			__memo(() => import('./nodes/6.js')),
			__memo(() => import('./nodes/7.js')),
			__memo(() => import('./nodes/8.js'))
		],
		routes: [
			{
				id: "/",
				pattern: /^\/$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 2 },
				endpoint: null
			},
			{
				id: "/contributions",
				pattern: /^\/contributions\/?$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 3 },
				endpoint: null
			},
			{
				id: "/dataset",
				pattern: /^\/dataset\/?$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 4 },
				endpoint: null
			},
			{
				id: "/methodology",
				pattern: /^\/methodology\/?$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 5 },
				endpoint: null
			},
			{
				id: "/problem",
				pattern: /^\/problem\/?$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 6 },
				endpoint: null
			},
			{
				id: "/references",
				pattern: /^\/references\/?$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 7 },
				endpoint: null
			},
			{
				id: "/results",
				pattern: /^\/results\/?$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 8 },
				endpoint: null
			}
		],
		prerendered_routes: new Set([]),
		matchers: async () => {
			
			return {  };
		},
		server_assets: {}
	}
}
})();
