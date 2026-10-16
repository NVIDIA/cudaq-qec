/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

// Version selector for the docs sidebar, rendered as the Read the Docs
// style flyout panel. sphinx_rtd_theme ships the CSS (.rst-versions et al.)
// and the expand/collapse handling for it even when not hosted on
// readthedocs.org, so this only needs to build the markup.
//
// The docs site is laid out as one directory per release version, with the
// newest release mirrored at the site root:
//
//   https://<host>/<site>/            <- mirror of the newest release
//   https://<host>/<site>/0.6.0/
//   https://<host>/<site>/0.7.0/
//
// The deployment workflow maintains a versions.json manifest at the site
// root, e.g. {"latest": "0.7.0", "versions": ["0.7.0", "0.6.0"]}. This
// script fetches the manifest and inserts the flyout. When switching
// versions it tries to stay on the current page, falling back to the target
// version's landing page if the page does not exist there.
(function () {
  "use strict";

  var script = document.currentScript;
  if (!script || !window.fetch) return;

  // .../<docs-root>/_static/version-switcher.js -> <docs-root>/
  var docRoot = new URL("../", script.src).href;

  function relativePagePath() {
    var page = new URL(window.location.href);
    var root = new URL(docRoot);
    if (page.pathname.indexOf(root.pathname) === 0) {
      return page.pathname.slice(root.pathname.length) + page.hash;
    }
    return "";
  }

  function navigateTo(version, siteRoot) {
    var targetRoot = new URL(version + "/", siteRoot).href;
    var samePage = targetRoot + relativePagePath();
    fetch(samePage, { method: "HEAD" })
      .then(function (response) {
        window.location.href = response.ok ? samePage : targetRoot;
      })
      .catch(function () {
        window.location.href = targetRoot;
      });
  }

  function insertFlyout(manifest, siteRoot) {
    var versions = manifest.versions || [];
    if (versions.length < 2) return;

    var current = null;
    for (var i = 0; i < versions.length; i++) {
      if (docRoot === new URL(versions[i] + "/", siteRoot).href) {
        current = versions[i];
      }
    }
    // Pages at the site root are a mirror of the newest release.
    if (current === null) current = manifest.latest;

    var flyout = document.createElement("div");
    flyout.className = "rst-versions";
    flyout.setAttribute("data-toggle", "rst-versions");
    flyout.setAttribute("role", "note");
    flyout.setAttribute("aria-label", "Documentation versions");

    var currentSpan = document.createElement("span");
    currentSpan.className = "rst-current-version";
    currentSpan.setAttribute("data-toggle", "rst-current-version");
    currentSpan.innerHTML =
        '<span class="fa fa-book"> CUDA-Q QEC</span> v: ' + current + " " +
        '<span class="fa fa-caret-down"></span>';
    flyout.appendChild(currentSpan);

    var others = document.createElement("div");
    others.className = "rst-other-versions";
    var dl = document.createElement("dl");
    var dt = document.createElement("dt");
    dt.textContent = "Versions";
    dl.appendChild(dt);

    versions.forEach(function (version) {
      var dd = document.createElement("dd");
      var link = document.createElement("a");
      // Plain link to the version root as the no-JS fallback; the click
      // handler upgrades it to a same-page switch when possible.
      link.href = new URL(version + "/", siteRoot).href;
      link.textContent = version;
      if (version === current) {
        var strong = document.createElement("strong");
        strong.appendChild(link);
        dd.appendChild(strong);
      } else {
        link.addEventListener("click", function (event) {
          event.preventDefault();
          navigateTo(version, siteRoot);
        });
        dd.appendChild(link);
      }
      dl.appendChild(dd);
    });

    others.appendChild(dl);
    flyout.appendChild(others);
    document.body.appendChild(flyout);
  }

  // The manifest lives at the site root: one level up when this page is
  // inside a version directory, or alongside the page for the root mirror.
  fetch(new URL("../versions.json", docRoot).href)
    .then(function (response) {
      if (!response.ok) throw new Error("no parent manifest");
      return response.json().then(function (manifest) {
        insertFlyout(manifest, new URL("../", docRoot).href);
      });
    })
    .catch(function () {
      return fetch(new URL("versions.json", docRoot).href)
        .then(function (response) {
          if (!response.ok) throw new Error("no manifest");
          return response.json();
        })
        .then(function (manifest) {
          insertFlyout(manifest, docRoot);
        })
        .catch(function () {
          /* No manifest (e.g. local build): no selector. */
        });
    });
})();
