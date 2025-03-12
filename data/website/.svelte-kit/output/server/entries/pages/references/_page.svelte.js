import { X as store_get, Z as unsubscribe_stores, V as pop, S as push } from "../../../chunks/index.js";
/* empty css                     */
import { a as attr, b as asideVisible } from "../../../chunks/stores.js";
function _page($$payload, $$props) {
  push();
  var $$store_subs;
  $$payload.out += `<section id="references"><h1>References</h1> <p>Here are the key references used in this project:</p> <ul><li>Heidari, M. et al. “Improving the performance of CNN to predict the likelihood of COVID-19 using chest X-ray images with preprocessing algorithms.” International Journal of Medical Informatics, 144 (2020): 104284. doi:10.1016/j.ijmedinf.2020.104284</li> <li>Jadwaa, S. K. “X-Ray Lung Image Classification Using a Canny Edge Detector.” Journal of Electrical and Computer Engineering, vol. 2022, 3081584, 8 pages, 2022. doi:10.1155/2022/3081584</li> <li>Rajpurkar, P. et al. “CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.” arXiv preprint arXiv:1711.05225 (2017). https://arxiv.org/abs/1711.05225</li></ul></section> <aside${attr("class", [
    !store_get($$store_subs ??= {}, "$asideVisible", asideVisible) ? "collapsed" : ""
  ].filter(Boolean).join(" "))}><div class="contents-header"><span class="header-text">On this page</span></div> <ul><li><a href="#references">References</a></li></ul></aside>`;
  if ($$store_subs) unsubscribe_stores($$store_subs);
  pop();
}
export {
  _page as default
};
