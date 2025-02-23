<script>
  import { onMount } from "svelte";
  import * as d3 from "d3";

  export let data = {};
  export let title = "";

  let container;
  let legendContainer;

  onMount(() => {
      if (!data || Object.keys(data).length === 0) return;
      drawStackedBarChart();
  });

  $: if (data && Object.keys(data).length > 0) {
      drawStackedBarChart();
  }

  function drawStackedBarChart() {
      d3.select(container).selectAll("*").remove();
      d3.select(legendContainer).html("");

      const width = 700;
      const height = 400;
      const margin = { top: 40, right: 30, bottom: 120, left: 80 };

      const conditions = Object.keys(data);
      const categories = ["Negative", "Positive", "Uncertain"];

      const stackData = conditions.map(condition => ({
          condition,
          ...data[condition]
      }));

      const x = d3.scaleBand()
          .domain(conditions)
          .range([0, width])
          .padding(0.3);

      const y = d3.scaleLinear()
          .domain([0, d3.max(stackData, d => d.Positive + d.Negative + d.Uncertain)])
          .nice()
          .range([height, 0]);

      const color = d3.scaleOrdinal()
          .domain(categories)
          .range(["#1f77b4", "#ff7f0e", "#2ca02c"]);

      const svg = d3.select(container)
          .append("svg")
          .attr("width", width + margin.left + margin.right)
          .attr("height", height + margin.top + margin.bottom)
          .append("g")
          .attr("transform", `translate(${margin.left},${margin.top})`);

      const stack = d3.stack().keys(categories);
      const stackedData = stack(stackData);

      const tooltip = d3.select("body").append("div")
          .attr("class", "tooltip")
          .style("opacity", 0);

      const groups = svg.selectAll("g.layer")
          .data(stackedData)
          .enter()
          .append("g")
          .attr("class", "layer")
          .attr("fill", d => color(d.key));

      groups.selectAll("rect")
          .data(d => d)
          .enter()
          .append("rect")
          .attr("x", d => x(d.data.condition))
          .attr("y", d => y(d[1]))
          .attr("height", d => y(d[0]) - y(d[1]))
          .attr("width", x.bandwidth())
          .on("mouseover", function (event, d) {
              tooltip.transition().duration(200).style("opacity", 1);
              tooltip.html(`${d.data.condition} - ${d3.select(this.parentNode).datum().key}: ${d[1] - d[0]}`)
                  .style("left", `${event.pageX + 10}px`)
                  .style("top", `${event.pageY - 20}px`);
          })
          .on("mouseout", function () {
              tooltip.transition().duration(200).style("opacity", 0);
          });

      svg.append("g")
          .attr("transform", `translate(0,${height})`)
          .call(d3.axisBottom(x))
          .selectAll("text")
          .attr("transform", "rotate(-45)")
          .style("text-anchor", "end");

      svg.append("g").call(d3.axisLeft(y));

      svg.append("text")
          .attr("text-anchor", "middle")
          .attr("x", width / 2)
          .attr("y", -10)
          .text(title)
          .style("font-size", "16px")
          .style("font-weight", "bold");

      const legend = svg.append("g")
          .attr("class", "legend")
          .attr("transform", `translate(${width / 2 - 170}, ${height + 80})`);

      const legendItems = legend.selectAll(".legend-item")
          .data(categories)
          .enter()
          .append("g")
          .attr("class", "legend-item")
          .attr("transform", (d, i) => `translate(${i * 120}, 0)`);

      legendItems.append("rect")
          .attr("width", 15)
          .attr("height", 15)
          .attr("fill", d => color(d));

      legendItems.append("text")
          .attr("x", 20)
          .attr("y", 12)
          .text(d => d)
          .style("font-size", "12px")
          .attr("alignment-baseline", "middle");
  }
</script>

<div class="chart-wrapper">
  <div bind:this={container} class="chart-svg"></div>
  <div bind:this={legendContainer} class="legend-container"></div>
</div>

<style>
  .chart-wrapper {
      display: flex;
      flex-direction: column;
      align-items: center;
      text-align: center;
      max-width: 750px;
      margin: auto;
  }

  .tooltip {
      position: absolute;
      background: rgba(0, 0, 0, 0.8);
      color: #fff;
      padding: 6px 10px;
      border-radius: 5px;
      font-size: 12px;
      pointer-events: none;
      transition: opacity 0.3s ease-in-out;
      z-index: 10;
  }

  .legend text {
    font-family: Arial, sans-serif;
    font-size: 12px;
    fill: #333;
  }

  .legend rect {
      stroke: #666;
      stroke-width: 1px;
  }

</style>
