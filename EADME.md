# Interactive Tabs with Markdown

Below is an example of an interactive tab-based content structure. Click on the tabs to view their respective content.

<!-- Tab Navigation -->
<div class="tabs">
  <button class="tab-link" onclick="openTab(event, 'Tab1')">Tab 1</button>
  <button class="tab-link" onclick="openTab(event, 'Tab2')">Tab 2</button>
  <button class="tab-link" onclick="openTab(event, 'Tab3')">Tab 3</button>
</div>

<!-- Tab Content -->
<div id="Tab1" class="tab-content">
  <h2>Tab 1 Content</h2>
  <p>This is the content for Tab 1. Add your details here.</p>
</div>

<div id="Tab2" class="tab-content" style="display:none;">
  <h2>Tab 2 Content</h2>
  <p>This is the content for Tab 2. Customize as needed.</p>
</div>

<div id="Tab3" class="tab-content" style="display:none;">
  <h2>Tab 3 Content</h2>
  <p>This is the content for Tab 3. You can include any information you'd like here.</p>
</div>

<!-- Style -->
<style>
  .tabs {
    display: flex;
    margin-bottom: 20px;
  }
  .tab-link {
    background-color: #f1f1f1;
    border: none;
    cursor: pointer;
    padding: 10px 20px;
    margin-right: 5px;
    font-size: 16px;
  }
  .tab-link:hover {
    background-color: #ddd;
  }
  .tab-link.active {
    background-color: #ccc;
  }
  .tab-content {
    padding: 20px;
    border: 1px solid #ddd;
    border-top: none;
  }
</style>

<!-- Script -->
<script>
  function openTab(event, tabId) {
    // Hide all tab content
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(content => (content.style.display = 'none'));

    // Remove active class from all buttons
    const tabLinks = document.querySelectorAll('.tab-link');
    tabLinks.forEach(link => link.classList.remove('active'));

    // Show the clicked tab's content and set the button as active
    document.getElementById(tabId).style.display = 'block';
    event.currentTarget.classList.add('active');
  }
</script>
