pipeline {
  agent { label "centos6 && dls" }

  options {
    skipDefaultCheckout(true)
  }

  parameters {
    booleanParam(defaultValue: false, description: 'Wipe the workspace and do a full build', name: 'FORCE_FULL_BUILD')
  }

  stages {
    stage('Bootstrap') {
      when {
        expression {
          !fileExists("base/.done") || params.FORCE_FULL_BUILD
        }
      }
      steps {
        deleteDir()
        sh "echo Branch ${env.BRANCH_NAME}"
        sh "env"
        checkout([$class: 'GitSCM', branches: [[name: BRANCH_NAME]],
          doGenerateSubmoduleConfigurations: false,
          extensions: [[$class: 'RelativeTargetDirectory', relativeTargetDir: 'modules/cctbx_project']],
          submoduleCfg: [], userRemoteConfigs: [[url: scm.userRemoteConfigs[0].url]]])
        sh "ln -s modules/cctbx_project/libtbx/auto_build/bootstrap.py ."
        sh "python bootstrap.py --builder=phenix --cciuser=ndevenish update hot"
        sh "python bootstrap.py --builder=phenix base"
      }
      post {
        failure {
          archiveArtifacts 'base_tmp/*_log'
        }
        success {
          sh "touch base/.done"
        }
      }
    }
    stage('Build') {
      steps {
        checkout([$class: 'GitSCM', branches: [[name: BRANCH_NAME]],
          doGenerateSubmoduleConfigurations: false,
          extensions: [[$class: 'RelativeTargetDirectory', relativeTargetDir: 'modules/cctbx_project']],
          submoduleCfg: [], userRemoteConfigs: [[url: scm.userRemoteConfigs[0].url  ]]])
        checkout([$class: 'GitSCM', branches: [[name: '*/master']],
          doGenerateSubmoduleConfigurations: false,
          extensions: [[$class: 'RelativeTargetDirectory', relativeTargetDir: 'modules/dials']],
          submoduleCfg: [], userRemoteConfigs: [[url: 'https://github.com/dials/dials.git']]])
        checkout([$class: 'GitSCM', branches: [[name: '*/master']],
          doGenerateSubmoduleConfigurations: false,
          extensions: [[$class: 'RelativeTargetDirectory', relativeTargetDir: 'modules/xia2_regression']],
          submoduleCfg: [], userRemoteConfigs: [[url: 'https://github.com/xia2/xia2_regression.git']]])
        // Relink the regression modules
        dir("modules") {
          sh "rm -f dials_regression && ln -s ${env.DIALS_REGRESSION} dials_regression"
          // sh "rm -f phenix_regression && ln -s ${env.PHENIX_REGRESSION} phenix_regression"
        }
        script {
          // Get the list of modules to configure for phenix
          modules = sh(returnStdout: true, script: "python -c 'import bootstrap; print \" \".join(bootstrap.PhenixBuilder.LIBTBX + bootstrap.PhenixBuilder.LIBTBX_EXTRA)'").trim()
          echo modules
          sh "mkdir -p ${env.WORKSPACE}/build && cd ${env.WORKSPACE}/build"
          dir("build") {
            if (fileExists("libtbx_env")) {
              // We have an existing build. Make sure we reconfigure. Make sure regression is included
            //   sh "cd ${env.WORKSPACE}/build && echo \"In \$(pwd)\" && ls && . ./setpaths.sh && libtbx.configure dials_regression"
              sh "./bin/libtbx.refresh"
            } else {
              // New configuration. include dials_regression and xia2_regression
              sh "../base/bin/python ../modules/cctbx_project/libtbx/configure.py ${modules} dials_regression xia2_regression"
            }
            sh "make"
          }
          sh "rsync -av /dls/science/groups/scisoft/DIALS/CD/build_dependencies/stash/xia2_regression_data/ ${env.WORKSPACE}/build/xia2_regression"
          sh "${env.WORKSPACE}/build/bin/mmtbx.rebuild_rotarama_cache"
        }
      }
    }
    stage('Test'){
      steps {
        // script {
        //   def dirExists(path) {sh(script: "[ -d '${path}' ]", returnStatus: true) == 0}
        // }
        dir(pwd(tmp: true)) {
          dir('_tests') {
            deleteDir()

            // sh ""
            // sh "source ${env.WORKSPACE}/build/setpaths.sh"
            script {
              PHENIX_REGRESSION = "${env.WORKSPACE}/modules/phenix_regression"
            }

            sh """module load ccp4 xds python/3.6.1 && \
            ${env.WORKSPACE}/build/bin/libtbx.run_tests_parallel nproc=15 \
              module=libtbx \
              module=scitbx \
              module=cctbx \
              module=iotbx \
              module=rstbx \
              module=dxtbx \
              module=dials \
              module=dials_regression \
              module=xia2 \
              module=xia2_regression \
              module=mmtbx \
              module=boost_adaptbx \
              module=smtbx \
              script='${PHENIX_REGRESSION}/ligand_pipeline/tst_pipeline_ligand_ncs.py' \
              script='${PHENIX_REGRESSION}/ligand_pipeline/tst_select_model.py' \
              script='${PHENIX_REGRESSION}/ligand_pipeline/tst_fo_minus_fo.py' \
              script='${PHENIX_REGRESSION}/ligand_pipeline/tst_no_ligand.py' \
              script='${PHENIX_REGRESSION}/ligand_pipeline/tst_modules.py' \
              directory='${PHENIX_REGRESSION}/libtbx' \
              directory='${PHENIX_REGRESSION}/scitbx' \
              directory='${PHENIX_REGRESSION}/cctbx' \
              directory='${PHENIX_REGRESSION}/iotbx' \
              directory='${PHENIX_REGRESSION}/mmtbx' \
              directory='${PHENIX_REGRESSION}/misc' \
              directory='${PHENIX_REGRESSION}/validation' \
              directory='${PHENIX_REGRESSION}/model_vs_data' \
              directory='${PHENIX_REGRESSION}/ligand' \
              directory='${PHENIX_REGRESSION}/xtriage' \
              directory='${PHENIX_REGRESSION}/gui' || true"""


            // sh '${WORKSPACE}/build/bin/libtbx.run_tests_parallel module=libtbx module=scitbx module=cctbx module=iotbx module=rstbx module=dxtbx module=dials module=xia2 nproc=3 output_junit_xml=True || true'
            junit 'output.xml'
          }
        }
      }
    }
  }
}
