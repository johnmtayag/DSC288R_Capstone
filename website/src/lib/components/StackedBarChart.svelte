<script>
  import { onMount } from "svelte";
  import * as d3 from "d3";

  // props
  export let data = {}; 
  export let title = ""; 

  let container;
  let legendSvgElement;
  let tooltipContainer;

  let svg;
  let xAxisGroup;
  let yAxisGroup;
  let tooltip;

  // layout
  const margin = { top: 40, right: 30, bottom: 70, left: 80 };
  const width = 700 - margin.left - margin.right;
  const height = 400 - margin.top - margin.bottom;
  const transitionDuration = 800;

  // categories & color scale
  const categories = ["Negative", "Positive", "Uncertain"];
  const colorScale = d3.scaleOrdinal()
    .domain(categories)
    .range(["#1f77b4", "#ff7f0e", "#2ca02c"]);

  // create SVG & axis groups
  onMount(() => {
    createChartStructure();
    createTooltip();

    if (data && Object.keys(data).length > 0) {
      updateChart();
    }
  });

  // when data changes, update chart again (if svg is ready)
  $: if (data && Object.keys(data).length > 0 && svg) {
    updateChart();
  }

  function createChartStructure() {
    // create overall <svg> with extra space for legend
    const mainSvg = d3.select(container)
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom + 50);

    // append a <g> for actual chart area
    svg = mainSvg.append("g")
      .attr("transform", `translate(${margin.left}, ${margin.top})`);

    // x-axis group at bottom
    xAxisGroup = svg.append("g")
      .attr("class", "x-axis")
      .attr("transform", `translate(0, ${height})`);

    // y-axis group at left
    yAxisGroup = svg.append("g")
      .attr("class", "y-axis");

    // chart title (append once, update text later)
    svg.append("text")
      .attr("class", "chart-title")
      .attr("text-anchor", "middle")
      .attr("x", width / 2)
      .attr("y", -10)
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text(title);
  }

  function createTooltip() {
    // create single tooltip div in absolute container
    tooltip = d3.select(tooltipContainer)
      .append("div")
      .attr("class", "bar-tooltip")
      .style("opacity", 0)
      .style("position", "absolute");
  }

  function updateChart() {
    // if empty data, do nothing
    const conditions = Object.keys(data);
    if (!conditions.length) return;

    // prepare data for stacking
    const processedData = conditions.map(condition => ({
      condition,
      ...data[condition],
      total: data[condition].Positive + data[condition].Negative + data[condition].Uncertain
    }));

    // setup scales
    const x = d3.scaleBand()
      .domain(processedData.map(d => d.condition))
      .range([0, width])
      .padding(0.3);

    const y = d3.scaleLinear()
      .domain([0, d3.max(processedData, d => d.total)])
      .nice()
      .range([height, 0]);

    // build stacked data for categories
    const stack = d3.stack().keys(categories);
    const stackedData = stack(processedData);

    // remove old bars before drawing new
    svg.selectAll(".stack-group").remove();

    // enter stack groups
    const groups = svg.selectAll("g.stack-group")
      .data(stackedData)
      .enter()
      .append("g")
      .attr("class", "stack-group")
      .attr("fill", d => colorScale(d.key));

    // enter rectangles in each group
    groups.selectAll("rect")
      .data(d => d, d => d.data.condition)
      .enter()
      .append("rect")
      .attr("x", d => x(d.data.condition))
      .attr("y", height)
      .attr("height", 0)
      .attr("width", x.bandwidth())
      .on("mouseover", function (event, d) {
        const stackKey = d3.select(this.parentNode).datum().key;
        const value = d[1] - d[0];
        updateTooltipContent(event, d.data, stackKey, value);
      })
      .on("mousemove", moveTooltip)
      .on("mouseleave", hideTooltip)
      // transition from y=bottom to stacked position
      .transition()
      .duration(transitionDuration)
      .attr("y", d => y(d[1]))
      .attr("height", d => y(d[0]) - y(d[1]));

    // x-axis
    xAxisGroup
      .transition()
      .duration(transitionDuration)
      .call(d3.axisBottom(x))
      .selectAll("text")
      .attr("transform", "rotate(-45)")
      .style("text-anchor", "end");

    // transition y-axis
    yAxisGroup
      .transition()
      .duration(transitionDuration)
      .call(d3.axisLeft(y));

    // update chart title (if `title` prop changes)
    svg.select(".chart-title").text(title);

    // rebuild legend
    d3.select(legendSvgElement).selectAll("*").remove();
    addLegend(d3.select(legendSvgElement));
  }

  function moveTooltip(event) {
    const [mouseX, mouseY] = d3.pointer(event, tooltipContainer);
    const containerRect = tooltipContainer.getBoundingClientRect();
    const containerWidth = containerRect.width;
    const containerHeight = containerRect.height;

    const tooltipRect = tooltip.node().getBoundingClientRect();
    const tooltipWidth = tooltipRect.width;
    const tooltipHeight = tooltipRect.height;

    let xPos = mouseX + 65; 
    let yPos = mouseY - 95;

    if (xPos + tooltipWidth > containerWidth) {
      xPos = mouseX - tooltipWidth + 35;
    }
    if (yPos + tooltipHeight > containerHeight) {
      yPos = mouseY - tooltipHeight - 10;
    }
    if (yPos < 0) {
      yPos = mouseY + 95;
    }
    if (xPos < 0) {
      xPos = mouseX + 65;
    }

    tooltip
      .style("left", `${xPos}px`)
      .style("top", `${yPos}px`);
  }

  function hideTooltip() {
    tooltip
      .style("opacity", 0)
      .style("transform", "translate(-50%, -100%) scale(0.9)");
  }

  function updateTooltipContent(event, d, stackKey, value) {
    tooltip
      .html(`
        <strong>${d.condition}</strong><br>
        <span style="color:${colorScale(stackKey)};">
          ${stackKey.toUpperCase()}
        </span><br>
        ${value} cases
      `)
      .style("opacity", 1)
      .style("transform", "translate(-50%, -100%) scale(1)");
  }

  function addLegend(legendSvg) {
    legendSvg
      .attr("width", width)
      .attr("height", 50);

    const legendGroup = legendSvg
      .append("g")
      .attr("class", "legend")
      .attr("transform", `translate(${width / 2 - 150}, 10)`);

    categories.forEach((key, i) => {
      const legendItem = legendGroup
        .append("g")
        .attr("class", "legend-item")
        .attr("transform", `translate(${i * 120},0)`);

      legendItem
        .append("rect")
        .attr("width", 15)
        .attr("height", 15)
        .attr("fill", colorScale(key))
        .style("cursor", "pointer")
        .on("click", () => {
          const stackGroup = svg.selectAll(`.stack-group[fill="${colorScale(key)}"]`);
          const currentOpacity = stackGroup.style("opacity");
          stackGroup
            .transition()
            .duration(transitionDuration)
            .style("opacity", currentOpacity == 1 ? 0 : 1);
        });

      legendItem
        .append("text")
        .attr("x", 20)
        .attr("y", 12)
        .attr("font-size", "12px")
        .attr("font-family", "Arial")
        .text(key);
    });
  }
</script>

<!-- DOM Layout -->
<div class="container">
  <!-- legend in a separate SVG -->
  <svg bind:this={legendSvgElement} class="legend-svg"></svg>
  
  <!-- main chart container -->
  <div bind:this={container} class="chart-svg"></div>
  
  <!-- tooltip container -->
  <div bind:this={tooltipContainer} class="tooltip-container"></div>
</div>

<!-- Styles -->
<style>
  .container {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    max-width: 750px;
    margin: auto;
  }

  .tooltip-container {
    position: relative;
    width: 100%;
  }

  :global(.bar-tooltip) {
    position: absolute;
    opacity: 0;
    background: rgba(0, 0, 0, 0.8);
    color: #fff;
    min-width: 115px;
    padding: 10px;
    border-radius: 5px;
    font-size: 12px;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    pointer-events: none;
    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.25);
    z-index: 1000;
    transition: opacity 0.5s ease, transform 0.5s ease;
    transform: translate(-50%, -100%) scale(0.9);
  }
</style>
