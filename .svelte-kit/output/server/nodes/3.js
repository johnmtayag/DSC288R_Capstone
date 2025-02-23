

export const index = 3;
let component_cache;
export const component = async () => component_cache ??= (await import('../entries/pages/contributions/_page.svelte.js')).default;
export const imports = ["_app/immutable/nodes/3.CqFWMRxw.js","_app/immutable/chunks/Bg9kRutz.js","_app/immutable/chunks/69_IOA4Y.js","_app/immutable/chunks/DIeogL5L.js"];
export const stylesheets = [];
export const fonts = [];
