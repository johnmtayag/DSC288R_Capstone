<script>
    import '$lib/global.css';
  </script>
  
  <section id="home">
    <h1>CheXpert Chest X-Ray Analysis</h1>
    <p>Welcome to the project showcase for our DSC 288R CheXpert Chest X-Ray Analysis.</p>
  </section>
  
  <section id="introduction">
    <h2>Overview</h2>
    <p>
      Chest radiography is a critical imaging technique used to diagnose many pathological conditions, but interpretation remains challenging, even for expert radiologists. The recent availability of large-scale, annotated X-ray datasets, such as CheXpert and MIMIC-CXR, has fueled interest in developing machine learning models for automated diagnosis.
    </p>
  </section>
  
  <section id="key-goals">
    <h2>Key Goals</h2>
    <p style="margin-bottom: 1rem;">However, uncertainty in model predictions and quality issues in training data pose significant challenges. Our project aims to:</p>
    <ul>
      <li>Improve classification accuracy and uncertainty analysis using pre-trained CNN architectures (e.g., DenseNet, ResNet).</li>
      <li>Leverage preprocessing techniques like histogram equalization and difference of Gaussians to enhance image quality.</li>
      <li>Use Grad-CAM visualizations to identify influential regions in X-ray images, enabling better interpretability.</li>
      <li>Incorporate uncertainty labels (-1) into the model pipeline to handle ambiguous cases.</li>
    </ul>
  </section>
  
  <aside>
    <div class="contents-header">On this page</div>
    <ul>
      <li><a href="#home">Home</a></li>
      <li><a href="#introduction">Introduction</a></li>
      <li><a href="#key-goals">Key Goals</a></li>
    </ul>
  </aside>