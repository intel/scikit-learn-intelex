fetch("https://aepanchi.github.io/scikit-learn-intelex/versions.json").then(response => response.text()).then(respText=>
    load_versions(respText));


function load_versions(json){
    var button = document.getElementById('version-switcher-button')
    var container = document.getElementById('version-switcher-dropdown')
    var loc = window.location.href;
    var s = document.createElement('select');
    s.style = "border-radius:5px;"
    const versions = JSON.parse(json);
    for (entry of versions){
        var o = document.createElement('option');
        var optionText = '';
        if ('name' in entry){
            optionText = entry.name;
        }else{
            optionText = entry.version;
        }
        o.value = entry.url;
        if (current_version == entry.version){
            o.selected = true;
            optionText = optionText;
        }
        o.innerHTML = optionText;
        s.append(o);
    }
    s.addEventListener("change", (event)=> {
        var current_url = new URL(window.location.href);
        var path = current_url.pathname;
        //strip version from path
        var page_path = path.substring(project_name.length+current_version.length+3);
        window.location.href = s.value + page_path;
    });
    container.append(s);
}