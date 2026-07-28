// ═══════════════════════════════════════════
// BALDES Landing — Scroll Reveal
// ═══════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    const reveals = document.querySelectorAll('.reveal');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -40px 0px'
    });

    reveals.forEach(el => observer.observe(el));

    // Nav background on scroll
    const nav = document.getElementById('main-nav');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            nav.style.background = 'rgba(11, 14, 17, 0.9)';
        } else {
            nav.style.background = 'rgba(11, 14, 17, 0.7)';
        }
    }, { passive: true });
});
