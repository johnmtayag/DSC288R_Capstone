import { V as pop, S as push, X as store_get, Z as unsubscribe_stores } from "../../../chunks/index.js";
/* empty css                     */
import "clsx";
import { a as attr, b as asideVisible } from "../../../chunks/stores.js";
function GradCAMOverylay($$payload, $$props) {
  push();
  let leftSubset, rightSubset;
  let allData = null;
  let hoveredConditionName = null;
  function getOverlayPath(subset, conditionName) {
    if (!subset || !subset.conditions) return null;
    const cond = subset.conditions.find((c) => c.name === conditionName);
    return cond ? cond.gradcam_path : null;
  }
  leftSubset = allData;
  rightSubset = allData;
  leftSubset?.conditions || rightSubset?.conditions || [];
  getOverlayPath(leftSubset, hoveredConditionName);
  getOverlayPath(rightSubset, hoveredConditionName);
  {
    $$payload.out += "<!--[!-->";
    $$payload.out += `<p class="loading svelte-q77p8">Loading metadata...</p>`;
  }
  $$payload.out += `<!--]-->`;
  pop();
}
function _page($$payload, $$props) {
  push();
  var $$store_subs;
  $$payload.out += `<section id="methodology"><h1>Methodology</h1></section> <section id="approach"><h2>Approach</h2> <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc id ipsum euismod, rhoncus orci at, varius diam. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Vivamus et ante turpis. Vivamus sit amet erat luctus, lacinia justo congue, lacinia eros. Phasellus rhoncus leo ac tortor aliquam, ac finibus augue luctus. Nam a blandit purus. Donec auctor semper nulla, ac consectetur enim sagittis quis. Proin tristique nisl tortor, euismod blandit lectus imperdiet id. Curabitur ac urna dignissim, eleifend lectus eu, tincidunt libero. Suspendisse potenti. Sed fermentum libero mollis, imperdiet sapien ac, dictum nisl.</p> <p style="margin-bottom: 1rem;">We train these models on two versions of the dataset:</p> <ol><li>Initial/Original chest X-rays (underwent some basic preprocessing?)</li> <li>Preprocessed images (using _______)</li></ol> <p style="margin-top: 1.5rem;">This allows us to analyze the impact of preprocessing on model accuracy and uncertainty.</p> `;
  GradCAMOverylay($$payload);
  $$payload.out += `<!----></section> <section id="success-criteria"><h2>Success Criteria</h2> <p>To quantify success, we will mainly utilize certain common machine learning metrics. We will analyze the accuracy of the models, both overall, but also between the different classes to identify any classes that the models struggle with. Considering this task involves automated diagnosis of serious pathological conditions, we will also analyze the sensitivity (true positive rate) and the specificity (true negative rate) of the models. As we wish to analyze the rate of uncertainty, we will also examine the fraction of instances either misclassified, or classified into the “Uncertain” class. With these metrics, we hope to identify both which classes the models struggle with the most, as well as any common sources of confusion found within the images themselves.</p></section> <aside${attr("class", [
    !store_get($$store_subs ??= {}, "$asideVisible", asideVisible) ? "collapsed" : ""
  ].filter(Boolean).join(" "))}><div class="contents-header"><span class="header-text">On this page</span></div> <ul><li><a href="#methodology">Methodology</a></li> <li><a href="#approach">Approach</a></li> <li><a href="#success-criteria">Success Criteria</a></li></ul></aside>`;
  if ($$store_subs) unsubscribe_stores($$store_subs);
  pop();
}
export {
  _page as default
};
