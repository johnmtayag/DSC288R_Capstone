<script>
  import { afterNavigate } from '$app/navigation';
  import BackToTop from '$lib/components/BackToTop.svelte';
  import { onMount } from 'svelte';
  import { base } from '$app/paths';

  const EXTRA_SPACE = 20;
  let topBar;

  // highlight main navigation link (e.g., Home, Problem, etc.)
  const highlightMainNav = () => {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('nav a');

    navLinks.forEach((link) => {
      link.classList.remove('active');
      if (link.getAttribute('href') === currentPath) {
        link.classList.add('active');
      }
    });
  };

  // highlight aside navigation link (e.g., #background, #motivation, etc.)
  const highlightAsideNav = () => {
    const sections = document.querySelectorAll('section');
    const asideLinks = document.querySelectorAll('aside ul li a');
    const topBarHeight = document.querySelector('.top-bar')?.offsetHeight || 0;

    let currentSection = '';

    sections.forEach((section) => {
      const sectionTop = section.offsetTop - topBarHeight - EXTRA_SPACE;
      const sectionHeight = section.offsetHeight;

      if (
        window.scrollY >= sectionTop - 10 &&
        window.scrollY < sectionTop + sectionHeight
      ) {
        currentSection = section.getAttribute('id');
      }
    });

    asideLinks.forEach((link) => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${currentSection}`) {
        link.classList.add('active');
      }
    });
  };

  // smooth scroll to section, accounting for the top bar height dynamically
  const smoothScrollToSection = (event, targetId) => {
    event.preventDefault();

    const targetSection = document.getElementById(targetId);
    const topBarHeight = document.querySelector('.top-bar')?.offsetHeight || 0;

    if (targetSection) {
      window.scrollTo({
        top: targetSection.offsetTop - topBarHeight - EXTRA_SPACE, // account for top bar height
        behavior: 'smooth',
      });
    }
  };

  // update highlighting for main and aside navigation links
  const updateHighlighting = () => {
    highlightMainNav();
    highlightAsideNav();
  };

  // attach listeners to aside links
  const attachAsideLinkListeners = () => {
    const asideLinks = document.querySelectorAll('aside ul li a');
    asideLinks.forEach((link) => {
      link.removeEventListener('click', handleLinkClick);
      link.addEventListener('click', handleLinkClick);
    });
  };

  // handle click event for aside links
  const handleLinkClick = (event) => {
    const targetId = event.target.getAttribute('href').substring(1);
    smoothScrollToSection(event, targetId);
  };

  // handle scroll event for top bar
  const handleScroll = () => {
    if (topBar) {
      if (window.scrollY > 0) {
        topBar.classList.add('scrolled');
      } else {
        topBar.classList.remove('scrolled');
      }
    }
  };

  // lifecycle: On mount
  onMount(() => {
    // initial highlighting
    updateHighlighting();

    // attach scroll listener for highlighting aside links
    window.addEventListener('scroll', highlightAsideNav);
    window.addEventListener('scroll', handleScroll);

    // attach listeners to aside links
    attachAsideLinkListeners();

    handleScroll();

    return () => {
      window.removeEventListener('scroll', highlightAsideNav);
      window.removeEventListener('scroll', handleScroll);
    };
  });

  // lifecycle: After navigation
  afterNavigate(() => {
    updateHighlighting(); // re-run highlighting when the route changes
    attachAsideLinkListeners(); // re-attach listeners to aside links after DOM updates
  });
</script>

<!-- Main Layout -->
<div bind:this={topBar} class="top-bar">
  <div class="top-bar-actions">
    <a href="https://github.com/johnmtayag/DSC288R_Capstone" target="_blank" class="icon-button" title="GitHub">
      <!-- GitHub SVG Icon -->
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="16" height="16">
        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.01.08-2.11 0 0 .67-.22 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.91.08 2.11.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38C13.71 14.53 16 11.54 16 8c0-4.42-3.58-8-8-8z"/>
      </svg>
    </a>
    <button class="icon-button" title="Download Options" aria-label="Download Options">⬇</button>
    <button class="icon-button" title="Fullscreen" aria-label="Fullscreen">⛶</button>
    <button class="icon-button" title="Toggle Dark Mode" aria-label="Toggle Dark Mode">🌗</button>
  </div>
</div>

<nav>
  <!-- header section -->
  <div class="nav-header">
    <h1>Project Title</h1>
    <!-- OR use an image -->
    <!-- <img src="/path/to/logo.png" alt="Logo" class="nav-logo" /> -->
  </div>

  <a href="{base}/" sveltekit:prefetch>Home</a>
  <a href="{base}/problem" sveltekit:prefetch>Problem</a>
  <a href="{base}/dataset" sveltekit:prefetch>Dataset</a>
  <a href="{base}/methodology" sveltekit:prefetch>Methodology</a>
  <a href="{base}/results" sveltekit:prefetch>Results</a>
  <!-- <a href="{base}/contributions" sveltekit:prefetch>Contributions</a> -->
  <a href="{base}/references" sveltekit:prefetch>References</a>
</nav>

<div class="content-wrapper">
  <aside>
    <slot name="aside">
      <!-- Default aside content -->
    </slot>
  </aside>

  <main>
    <slot />
  </main>
</div>

<BackToTop />