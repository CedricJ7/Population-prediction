// Animation des statistiques au chargement
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(function() {
        const statCards = document.querySelectorAll('.stat-card p');
        statCards.forEach(card => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(20px)';
            card.style.transition = 'opacity 0.8s ease-out, transform 0.8s ease-out';
            
            setTimeout(function() {
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, 100);
        });
    }, 300);
});

// Amélioration des tooltips au survol
document.addEventListener('mouseover', function(e) {
    if (e.target.closest('.js-plotly-plot')) {
        const tooltips = document.querySelectorAll('.hovertext');
        tooltips.forEach(tooltip => {
            tooltip.style.boxShadow = '0 4px 20px rgba(0,0,0,0.15)';
            tooltip.style.borderRadius = '6px';
        });
    }
});

// Ajouter cette fonction pour masquer l'écran de chargement
window.addEventListener('load', function() {
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (loadingOverlay) {
        setTimeout(function() {
            loadingOverlay.style.opacity = '0';
            setTimeout(function() {
                loadingOverlay.style.display = 'none';
            }, 500);
        }, 800);
    }
});

// Ajouter ceci pour les transitions douces entre onglets
document.addEventListener('DOMContentLoaded', function() {
    // Surveiller les changements d'onglets
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList' && mutation.addedNodes.length) {
                const addedContent = Array.from(mutation.addedNodes).find(node => 
                    node.nodeType === Node.ELEMENT_NODE && 
                    (node.id === 'main-content' || node.querySelector('#main-content'))
                );
                
                if (addedContent) {
                    // Appliquer une transition d'entrée au contenu
                    const content = addedContent.id === 'main-content' ? addedContent : addedContent.querySelector('#main-content');
                    if (content) {
                        content.style.opacity = '0';
                        content.style.transform = 'translateY(20px)';
                        
                        setTimeout(function() {
                            content.style.transition = 'opacity 0.4s ease-out, transform 0.4s ease-out';
                            content.style.opacity = '1';
                            content.style.transform = 'translateY(0)';
                        }, 50);
                    }
                }
            }
        });
    });
    
    // Observer le conteneur principal
    observer.observe(document.querySelector('.main-content'), { childList: true, subtree: true });
});

// Amélioration des tableaux
document.addEventListener('DOMContentLoaded', function() {
    // Mettre en surbrillance les lignes du tableau au survol
    document.addEventListener('mouseover', function(e) {
        if (e.target.tagName === 'TD') {
            const row = e.target.closest('tr');
            if (row) {
                row.style.backgroundColor = '#EEF5FD';
            }
        }
    });
    
    document.addEventListener('mouseout', function(e) {
        if (e.target.tagName === 'TD') {
            const row = e.target.closest('tr');
            if (row) {
                row.style.backgroundColor = '';
            }
        }
    });
}); 