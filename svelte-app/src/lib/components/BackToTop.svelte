<script>
    import { onMount } from 'svelte';
    
    let isVisible = false;
  
    function handleScroll() {
      isVisible = window.scrollY > 500;
    }
  
    function scrollToTop() {
      window.scrollTo({
        top: 0,
        behavior: 'smooth',
      });
    }
  
    onMount(() => {
      window.addEventListener('scroll', handleScroll);
  
      return () => {
        window.removeEventListener('scroll', handleScroll);
      };
    });
  </script>
  
  <style>
    .back-to-top {
        position: fixed;
        bottom: 30px;
        right: 30px;
        background-color: #fff;
        color: #333;
        padding: 10px 15px;
        font-size: 16px;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transition: opacity 0.3s ease, transform 0.3s ease;
        opacity: 0;
        visibility: hidden;
        z-index: 1000;
    }
  
    .back-to-top.visible {
        opacity: 1;
        visibility: visible;
        transform: translateY(0);
    }
  
    .back-to-top:hover {
        background-color: #333;
        color: #fff;
        transform: translateY(-5px);
    }
  </style>
  
  <button
    class="back-to-top {isVisible ? 'visible' : ''}"
    on:click={scrollToTop}
    aria-label="Back to Top"
  >
    ⤴
  </button>  