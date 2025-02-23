export const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "test/_app",
	assets: new Set(["favicon.png","summary.json"]),
	mimeTypes: {".png":"image/png",".json":"application/json"},
	_: {
		client: {start:"_app/immutable/entry/start.B6kKZZjO.js",app:"_app/immutable/entry/app.D5EQoJzK.js",imports:["_app/immutable/entry/start.B6kKZZjO.js","_app/immutable/chunks/HxPqkZ8F.js","_app/immutable/chunks/eP3un58F.js","_app/immutable/chunks/Bm0FhLEK.js","_app/immutable/chunks/DIeogL5L.js","_app/immutable/entry/app.D5EQoJzK.js","_app/immutable/chunks/Bm0FhLEK.js","_app/immutable/chunks/DIeogL5L.js","_app/immutable/chunks/C92lvqdm.js","_app/immutable/chunks/CHZwxh0T.js","_app/immutable/chunks/BQPxy4xd.js","_app/immutable/chunks/Bg9kRutz.js","_app/immutable/chunks/F2zvKX18.js","_app/immutable/chunks/Cy-pn9s0.js","_app/immutable/chunks/eP3un58F.js"],stylesheets:[],fonts:[],uses_env_dynamic_public:false},
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
