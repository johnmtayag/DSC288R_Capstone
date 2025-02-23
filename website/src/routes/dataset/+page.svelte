<script>
    import '$lib/global.css';
    import { onMount } from "svelte";
    import PieChart from "$lib/components/PieChart.svelte";
    import StackedBarChart from "$lib/components/StackedBarChart.svelte";
    import { base } from '$app/paths';

    let data = null;
    let selectedDataset = "filtered"; // default to filtered dataset


    onMount(async () => {
        try {
            const response = await fetch(`${base}/summary.json`);
            data = await response.json();
        } catch (err) {
            console.error("Error loading summary.json:", err);
        }
    });
  </script>

  <section id="dataset">
    <h1>Dataset and Features</h1>
  </section>

  <section id="overview">
    <h2>Overview</h2>
    <p>The CheXpert dataset is a large-scale collection of 224,316 chest X-ray images from 65,240 patients, designed to aid in automated radiographic interpretation. Chest X-ray diagnosis is often challenging, even for expert radiologists, due to the complexity of pathological patterns. Our goal is to analyze uncertainty in classification models by examining factors such as image resolution and quality.
    </p>
    <p>
      For this study, we use a 90GB subset of the dataset to balance computational feasibility with model performance. This subset contains images labeled using CheXbert, a BERT-based natural language processing model, while validation and test sets use radiologist-labeled data, ensuring a strong ground truth comparison.
    </p>

    <section id="dataset-composition">
      <h2>Dataset Composition</h2>
      <p style="margin-bottom: 1rem;">
        The dataset is divided into three primary subsets:
      </p>
        <ul>
          <li>Training Data: Labeled using the CheXbert NLP model</li>
          <li>Validation Data: Labeled by three board-certified radiologists</li>
          <li>Test Data: Labeled by five board-certified radiologists</li>
        </ul>

        <p style="margin-top: 1.5rem; margin-bottom: 1rem" m>Link to dataset sources:
        </p>
          <ul>
            <li><a href="https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2">CheXpert (Training & Validation Data)</a></li>
            <li><a href="https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2">CheXlocalize (Test Data)</a></li>
          </ul>
    </section>

    <section id="image-characteristics">
      <h2>Image Characteristics</h2>
      <p>
        Frontal vs. Lateral Views: The dataset contains predominantly frontal chest X-rays, as they are the clinical standard for diagnosis. Lateral views were excluded to maintain consistency.
      </p>
      <p style="margin-bottom: 1rem;">
        <b>AP vs PA Views:</b>
      </p>
        <ul>
          <li><b>Anteroposterior (AP) views</b> are taken when a patient is lying down or unable to stand.</li>
          <li><b>Posteroanterior (PA) views</b> are the preferred clinical standard but are less frequent in the dataset due to the hospital-based nature of CheXpert.</li>
        </ul>
    </section>

      {#if data}
          <div class="chart-row">
              <PieChart title="Frontal vs Lateral" data={data.view_distribution.original} />
              <PieChart title="AP vs PA Projection" data={data.projection_distribution.original} />
          </div>

        <div class="dataset-toggle">
          <label for="datasetSelector">Select Dataset: </label>
          <select id="datasetSelector" bind:value={selectedDataset}>
              <option value="filtered">Filtered Dataset (Frontal AP only)</option>
              <option value="original">Original Dataset</option>
          </select>
      </div>

      <StackedBarChart title="Label Distribution Per Condition" data={data.label_distribution[selectedDataset]} />
    {:else}
        <p>Loading data...</p>
    {/if}
</section>
  
  <section id="preprocessing">
    <h2>Preprocessing Steps</h2>
    <p style="margin-bottom: 1rem;"> To standardize and clean the dataset, we applied the following steps:</p>  
        <ol>
            <li>Resizing all images to a uniform resolution (three versions: 224×224, 384×384, 512×512).</li>
            <li>Handling missing values: Since the dataset uses "N/A" to indicate conditions not mentioned in radiology reports, we map them to negative labels (assuming no mention implies absence).</li>
            <li>Filtering out low-quality images: Images with severe artifacts, extreme distortions, or blank scans were removed.</li>
        </ol>
  </section>

  <section id="eda">
    <h2>Exploratory Data Analysis (EDA)</h2>
    <p style="margin-bottom: 1rem;"> To standardize and clean the dataset, we applied the following steps:</p>  
        <ol>
            <li>Resizing all images to a uniform resolution (three versions: 224×224, 384×384, 512×512).</li>
            <li>Handling missing values: Since the dataset uses "N/A" to indicate conditions not mentioned in radiology reports, we map them to negative labels (assuming no mention implies absence).</li>
            <li>Filtering out low-quality images: Images with severe artifacts, extreme distortions, or blank scans were removed.</li>
        </ol>
  </section>

  <section id="success-criteria">
    <h2>Success Criteria</h2>
    <p>
        To quantify success, we will mainly utilize certain common machine learning metrics. We will analyze the accuracy of the models, both overall, but also between the different classes to identify any classes that the models struggle with. Considering this task involves automated diagnosis of serious pathological conditions, we will also analyze the sensitivity (true positive rate) and the specificity (true negative rate) of the models. As we wish to analyze the rate of uncertainty, we will also examine the fraction of instances either misclassified, or classified into the “Uncertain” class. With these metrics, we hope to identify both which classes the models struggle with the most, as well as any common sources of confusion found within the images themselves.
    </p>
  </section>
  
  <aside>
    <div class="contents-header">On this page</div>
    <ul>
      <li><a href="#overview">Overview</a></li>
      <li><a href="#dataset-composition">Dataset Composition</a></li>
      <li><a href="#image-characteristics">Image Characteristics</a></li>
      <li><a href="#preprocessing">Preprocessing Steps</a></li>
      <li><a href="#success-criteria">Success Criteria</a></li>
    </ul>
  </aside>

  <style>
    .chart-row {
        display: flex;
        justify-content: center;
        gap: 30px;
        flex-wrap: wrap;
        margin-top: 1.2rem;
    }

    .dataset-toggle {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 20px 0;
        font-size: 16px;
    }

    .dataset-toggle select {
        margin-left: 10px;
        padding: 5px;
        font-size: 16px;
    }
</style>