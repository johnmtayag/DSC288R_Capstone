import { V as pop, W as stringify, S as push, X as store_get, Y as slot, Z as unsubscribe_stores } from "../../chunks/index.js";
import "../../chunks/client.js";
import { a as attr, b as asideVisible } from "../../chunks/stores.js";
import { b as base } from "../../chunks/paths.js";
function BackToTop($$payload, $$props) {
  push();
  $$payload.out += `<button${attr("class", `back-to-top ${stringify("")} svelte-gqkr8v`)} aria-label="Back to Top">⤴</button>`;
  pop();
}
function _layout($$payload, $$props) {
  push();
  var $$store_subs;
  $$payload.out += `<div${attr("class", `top-bar ${stringify([
    !store_get($$store_subs ??= {}, "$asideVisible", asideVisible) ? "expanded" : ""
  ].filter(Boolean).join(" "))}`)}><div class="top-bar-actions"><a href="https://github.com/johnmtayag/DSC288R_Capstone" target="_blank" class="icon-button" title="GitHub"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="16" height="16"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.01.08-2.11 0 0 .67-.22 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.91.08 2.11.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38C13.71 14.53 16 11.54 16 8c0-4.42-3.58-8-8-8z"></path></svg></a> <button class="icon-button" title="Download Options" aria-label="Download Options">⬇</button> <button class="icon-button" title="Fullscreen" aria-label="Fullscreen">⛶</button> <button class="icon-button" title="Toggle Dark Mode" aria-label="Toggle Dark Mode">🌗</button></div></div> <nav><div class="nav-header"><h1>Project Title</h1></div> <a${attr("href", `${stringify(base)}/`)} sveltekit:prefetch="">Home</a> <a${attr("href", `${stringify(base)}/problem`)} sveltekit:prefetch="">Problem</a> <a${attr("href", `${stringify(base)}/dataset`)} sveltekit:prefetch="">Dataset</a> <a${attr("href", `${stringify(base)}/methodology`)} sveltekit:prefetch="">Methodology</a> <a${attr("href", `${stringify(base)}/results`)} sveltekit:prefetch="">Results</a> <a${attr("href", `${stringify(base)}/references`)} sveltekit:prefetch="">References</a></nav> <div class="content-wrapper"><aside${attr("class", [
    !store_get($$store_subs ??= {}, "$asideVisible", asideVisible) ? "collapsed" : ""
  ].filter(Boolean).join(" "))}><!---->`;
  slot($$payload, $$props, "aside", {}, () => {
  });
  $$payload.out += `<!----></aside> <main${attr("class", [
    !store_get($$store_subs ??= {}, "$asideVisible", asideVisible) ? "expanded" : ""
  ].filter(Boolean).join(" "))}><!---->`;
  slot($$payload, $$props, "default", {}, null);
  $$payload.out += `<!----></main></div> `;
  BackToTop($$payload);
  $$payload.out += `<!---->`;
  if ($$store_subs) unsubscribe_stores($$store_subs);
  pop();
}
export {
  _layout as default
};
